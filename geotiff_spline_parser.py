import numpy as np
import rasterio
import torch
from scipy.interpolate import RectBivariateSpline

class GeoTIFFSplineParser:
    def __init__(self, tiff_path):
        """
        Загружает реальный GeoTIFF цифровой модели рельефа (DEM) 
        и строит непрерывный кубический сплайн.
        """
        self.tiff_path = tiff_path
        
        # 1. Читаем растровые геоданные средствами rasterio
        with rasterio.open(self.tiff_path) as src:
            # Читаем первую матрицу высот (band 1)
            self.dem_matrix = src.read(1).astype(np.float32)
            
            # Извлекаем метаданные разрешения сетки (размер пикселя в метрах)
            self.transform = src.transform
            self.dx = self.transform[0] # Шаг по X (например, 30 метров для SRTM)
            self.dy = abs(self.transform[4]) # Шаг по Y
            
            self.height, self.width = self.dem_matrix.shape
            
        # 2. Создаем сетку координат для Scipy
        self.x_coords = np.arange(0, self.width) * self.dx
        self.y_coords = np.arange(0, self.height) * self.dy
        
        print(f"🌍 Растр загружен: {self.width}x{self.height} пикселей.")
        print(f"   └── Разрешение сетки: DX={self.dx}м, DY={self.dy}м")
        print(f"   └── Минимальная высота: {np.min(self.dem_matrix):.1f}м, Максимальная: {np.max(self.dem_matrix):.1f}м")
        
        # 3. СТРОИМ ДВУМЕРНЫЙ КУБИЧЕСКИЙ СПЛАЙН (Магия сглаживания)
        # kx=3, ky=3 означает кубический сплайн (дает гладкую 1-ю и 2-ю производные)
        print("🧮 Построение непрерывного 2D кубического сплайна...")
        self.spline = RectBivariateSpline(self.x_coords, self.y_coords, self.dem_matrix.T, kx=3, ky=3)
        print("🎉 Сплайн готов. Рельеф Непала переведен в непрерывное математическое поле.")

    def get_height_and_slopes_tensor(self, x_tensor, y_tensor):
        """
        Принимает тензоры координат PyTorch и возвращает физический уклон (slope_x, slope_y)
        вычисленный через аналитическую производную сплайна.
        """
        # Переводим тензоры PyTorch во временные массивы CPU для Scipy
        x_np = x_tensor.detach().cpu().numpy().flatten()
        y_np = y_tensor.detach().cpu().numpy().flatten()
        
        # Вычисляем высоты
        z_values = self.spline(x_np, y_np, grid=False).astype(np.float32)
        
        # 🛠️ СЧИТАЕМ АНАЛИТИЧЕСКИЕ ПРОИЗВОДНЫЕ СПЛАЙНА (dx=1, dy=1 означает первую производную)
        dz_dx = self.spline(x_np, y_np, dx=1, dy=0, grid=False).astype(np.float32)
        dz_dy = self.spline(x_np, y_np, dx=0, dy=1, grid=False).astype(np.float32)
        
        # Возвращаем данные обратно в тензоры PyTorch на тот же девайс (CPU/CUDA)
        device = x_tensor.device
        z_torch = torch.from_numpy(z_values).unsqueeze(1).to(device)
        slope_x = torch.from_numpy(dz_dx).unsqueeze(1).to(device)
        slope_y = torch.from_numpy(dz_dy).unsqueeze(1).to(device)
        
        return z_torch, slope_x, slope_y

# Пример быстрой интеграции в наше ИИ-ядро PINN:
if __name__ == "__main__":
    # Для демонстрации создаем фейковый .tif файл высот Лангтанг
    # В реальности вы скачиваете DEM-файл (например, SRTM или ALOS PALSAR)
    print("=== Тестирование сплайн-парсера на реальной математике ===")
    
    # Генерируем тестовую матрицу высот (имитация ущелья Ленде)
    synthetic_nepal_dem = np.zeros((256, 256), dtype=np.float32)
    for i in range(256):
        for j in range(256):
            synthetic_nepal_dem[i, j] = 5200.0 - j * 12.0 + np.sin(i/10.0) * 50.0 # Спуск с 5200м до 2100м
            
    # Сохраняем как временный файл растра
    with rasterio.open(
        'nepal_langtang_dem.tif', 'w', driver='GTiff',
        height=256, width=256, count=1, dtype='float32',
        crs='+proj=latlong',
        transform=rasterio.transform.from_origin(85.525, 28.285, 30.0, 30.0) # Реальные координаты и шаг 30 метров
    ) as dst:
        dst.write(synthetic_nepal_dem, 1)

    # Инициализируем наш парсер
    parser = GeoTIFFSplineParser('nepal_langtang_dem.tif')
    
    # Проверяем, как авторад PyTorch получит гладкие уклоны в случайных точках ущелья
    test_x = torch.rand(5, 1) * 2000.0 # Случайные метры по X
    test_y = torch.rand(5, 1) * 2000.0 # Случайные метры по Y
    
    z, sx, sy = parser.get_height_and_slopes_tensor(test_x, test_y)
    
    print("\n🎯 Проверка тензоров PyTorch для ИИ:")
    for i in range(5):
        print(f"Точка {i}: Координаты ({test_x[i].item():.1f}м, {test_y[i].item():.1f}м) -> Высота горы: {z[i].item():.1f}м | Уклон X: {sx[i].item():.4f}, Уклон Y: {sy[i].item():.4f}")
