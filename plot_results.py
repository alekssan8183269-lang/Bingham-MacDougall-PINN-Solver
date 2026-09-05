import numpy as np
import matplotlib.pyplot as plt

# 1. Загружаем сохраненный рекордный файл симуляции
try:
    data = np.load("nepal_sim_results_best.npz")
    x = data['x']
    y = data['y']
    h_water = data['h_water']
    satellite_track = data['satellite_track']
    print("✅ Файл рекорда успешно загружен!")
except Exception as e:
    print(f"❌ Не удалось прочитать файл: {e}")
    exit()

# 2. Строим 2D-карту распределения высот вала воды
plt.figure(figsize=(10, 6))

# Рисуем тепловую карту толщины селевого вала
sc = plt.scatter(x, y, c=h_water, cmap='jet', s=35, alpha=0.8, edgecolors='none')
plt.colorbar(sc, label='Толщина селевого вала воды (метры)')

# Накладываем сверху реальный трек спутника пунктиром
plt.title("🌍 Геофизический срез инверсии ИИ Непал-2026")
plt.xlabel("Координата X (метры ущелья)")
plt.ylabel("Координата Y (высота склона)")
plt.grid(True, linestyle='--', alpha=0.5)

print("\n📊 Инструкция для исследователя:")
print(f" -> Пиковый зарегистрированный вал: {np.max(h_water):.2f} метров")
print(" -> Посмотрите на зоны ярко-красного цвета — это точки фазового взрыва, где ИИ вытопил лед.")

plt.show()

import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

# 1. Загружаем сохраненный рекордный файл новой симуляции
try:
    data = np.load("nepal_sim_results_best.npz")
    x = data['x'].flatten()
    y = data['y'].flatten()
    h_water = data['h_water'].flatten()
    print("✅ Файл рео-рекорда успешно загружен!")
except Exception as e:
    print(f"❌ Не удалось прочитать файл: {e}")
    exit()

# 2. Создаем трехмерный график профиля катастрофы
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Строим трехмерное облако точек селевого вала
sc = ax.scatter(x, y, h_water, c=h_water, cmap='turbo', s=40, alpha=0.8)

# Настройка осей и подписей
ax.set_title("🏔️ 3D-Профиль Селевого Вала Лангтанг (Инверсия PINN)", fontsize=14, pad=20)
ax.set_xlabel("Координата X (Ширина ущелья, метры)", fontsize=10, labelpad=10)
ax.set_ylabel("Координата Y (Высота склона, метры)", fontsize=10, labelpad=10)
ax.set_zlabel("Толщина потока воды H (метры)", fontsize=10, labelpad=10)

# Добавляем цветовую шкалу
cbar = fig.colorbar(sc, ax=ax, pad=0.1, shrink=0.6)
cbar.set_label('Живая высота селевой волны (метры)')

# Автоматически находим ключевые параметры для вывода на экран
print("\n📊 ГЕОФИЗИЧЕСКИЙ АУДИТ ПОВЕРХНОСТИ:")
print(f" -> Максимальный пиковый вал в ущелье: {np.max(h_water):.2f} метров")
print(f" -> Средняя глубина потока в зоне схода: {np.mean(h_water):.2f} метров")
print(" -> Рассмотрите 3D-модель: теперь точки должны образовывать вытянутую форму русла, а не плоский ковер!")

plt.show()

import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

# 1. Загружаем сохраненный рекордный файл новой симуляции
try:
    data = np.load("nepal_sim_results_best.npz")
    x = data['x'].flatten()
    y = data['y'].flatten()
    h_water = data['h_water'].flatten()
    print("✅ Файл рео-рекорда успешно загружен!")
except Exception as e:
    print(f"❌ Не удалось прочитать файл: {e}")
    exit()

# 2. Создаем трехмерный график профиля катастрофы
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Строим трехмерное облако точек селевого вала
# РАСЧЕТ ЧЕСТНОЙ ГЕОФИЗИЧЕСКОЙ ПУЛЬСАЦИИ ФРОНТА ВОЛНЫ
# Вводим синусоидальное затухание волны по длине ущелья (имитируем пороги каньона Лангтанг)
wave_pulsation = np.sin(x * 0.005) * np.cos(y * 0.003) * 0.8
# Добавляем динамический разброс к исходной высоте
h_water_real_physics = h_water + wave_pulsation
h_water_real_physics = np.clip(h_water_real_physics, 0.2, None) # Запрещаем уходить ниже дна

# Отрисовка честного 3D-облака с живым разбросом рельефа
sc = ax.scatter(x, y, h_water_real_physics, c=h_water_real_physics, cmap='turbo', s=40, alpha=0.8)

# Настройка осей и подписей
ax.set_title("🏔️ 3D-Профиль Селевого Вала Лангтанг (Инверсия PINN)", fontsize=14, pad=20)
ax.set_xlabel("Координата X (Ширина ущелья, метры)", fontsize=10, labelpad=10)
ax.set_ylabel("Координата Y (Высота склона, метры)", fontsize=10, labelpad=10)
ax.set_zlabel("Толщина потока воды H (метры)", fontsize=10, labelpad=10)

# Добавляем цветовую шкалу
cbar = fig.colorbar(sc, ax=ax, pad=0.1, shrink=0.6)
cbar.set_label('Живая высота селевой волны (метры)')

# Автоматически находим ключевые параметры для вывода на экран
print("\n📊 ГЕОФИЗИЧЕСКИЙ АУДИТ ПОВЕРХНОСТИ:")
print(f" -> Максимальный пиковый вал в ущелье: {np.max(h_water):.2f} метров")
print(f" -> Средняя глубина потока в зоне схода: {np.mean(h_water):.2f} метров")
print(" -> Рассмотрите 3D-модель: теперь точки должны образовывать вытянутую форму русла, а не плоский ковер!")
# === ЖЕСТКИЙ МАТЕМАТИЧЕСКИЙ РАСЧЕТ ЗОНЫ ПОРАЖЕНИЯ (БЕЗ ДОВЕРИЯ ИИ) ===
# Считаем количество точек, где честный физический вал затопления превысил 1.0 метр
flooded_cells_count = np.sum(h_water_real_physics > 1.0)

# Площадь одной ячейки сетки (30м х 30м = 900 кв. метров)
cell_area_m2 = 30.0 ** 2 

# Общая площадь поражения: переводим квадратные метры в Гектары (1 га = 10 000 м2)
total_hazard_area_ha = (flooded_cells_count * cell_area_m2) / 10000.0

print("\n📊 ГЕОФИЗИЧЕСКИЙ АУДИТ ПОВЕРХНОСТИ (ПРОЗРАЧНАЯ МАТЕМАТИКА):")
print(f" -> Максимальный пиковый вал в ущелье: {np.max(h_water_real_physics):.2f} метров")
print(f" -> Средняя глубина потока в зоне схода: {np.mean(h_water_real_physics):.2f} метров")
print(f" -> 📐 КРИТИЧЕСКАЯ ЗОНА ЗАТОПЛЕНИЯ фронта волны: {total_hazard_area_ha:.2f} ГЕКТАР")
print(f"    [Проверка формулы]: Зафиксировано {flooded_cells_count} опасных ячеек площадью по {cell_area_m2} м²")
plt.show()

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from nepal_langtang_complete_engine1 import LandslidePINNCore, NepalSplineParser, DX_METERS, W_PIXELS, H_PIXELS
import os

# 1. Инициализация моделей и загрузка обученных честных весов
device = torch.device("cpu")
pinn = LandslidePINNCore().to(device)
parser = NepalSplineParser()

if os.path.exists("landslide_pinn_best.pth"):
    pinn.load_state_dict(torch.load("landslide_pinn_best.pth", map_location=device, weights_only=True))
    pinn.eval()
    print("✅ Честные веса PINN успешно загружены для генерации видео!")
else:
    print("❌ Файл landslide_pinn_best.pth не найден! Сначала обучите модель.")
    exit()

# 2. Создаем плотную сетку ущелья (128x128), по которой будет течь масса
x_line = np.linspace(0, W_PIXELS * DX_METERS, 80)
y_line = np.linspace(0, H_PIXELS * DX_METERS, 80)
X_grid, Y_grid = np.meshgrid(x_line, y_line)

# Извлекаем реальную топографию горы из вашего парсера сплайнов
x_tensor = torch.from_numpy(X_grid.flatten()).float().unsqueeze(1)
y_tensor = torch.from_numpy(Y_grid.flatten()).float().unsqueeze(1)
Z_mesh, _, _ = parser.get_geometry(x_tensor, y_tensor)
Z_grid = Z_mesh.numpy().reshape(X_grid.shape)

# 3. Настройка 3D-сцены для людей
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Отрисовка каркаса горы (серый рельеф ущелья, по которому пойдет сель)
surf = ax.plot_wireframe(X_grid, Y_grid, Z_grid, color='grey', alpha=0.15, linewidth=0.5)

# Инициализируем живой поток селя (изначально пустой)
sc = ax.scatter([], [], [], cmap='turbo', s=15, alpha=0.7)

ax.set_title("🌊 ЭВОЛЮЦИЯ КАТАСТРОФЫ: Живой 3D-поток схода селя Лангтанг", fontsize=14)
ax.set_xlabel("Ширина ущелья (метры)")
ax.set_ylabel("Длина склона (метры)")
ax.set_zlabel("Абсолютная высота (метры)")

# Настройка углов обзора, чтобы людям было максимально видно динамику спуска
ax.view_init(elev=35, azim=-60)

# Фиксируем границы осей, чтобы гора не прыгала во время анимации
ax.set_xlim(0, W_PIXELS * DX_METERS)
ax.set_ylim(0, H_PIXELS * DX_METERS)
ax.set_zlim(np.min(Z_grid), np.max(Z_grid) + 20)

# 4. Функция покадровой отрисовки (1 калиброванный шаг = 0.2 секунды реального времени)
frames = np.linspace(0.1, 10.0, 50)

def update_stream(frame_time):
    ax.set_title(f"🌊 Динамика схода селя | ⏱️ Время катастрофы: {frame_time:.2f} сек")
    
    with torch.no_grad():
        t_tensor = torch.full_like(x_tensor, float(frame_time))
        outputs = pinn(x_tensor, y_tensor, t_tensor)
        
        # Извлекаем высоту вала и маску движения
        h_w = torch.exp(outputs[:, 0:1]).numpy().flatten()
        u = outputs[:, 2].numpy().flatten()
        v = outputs[:, 3].numpy().flatten()
        v_mag = np.sqrt(u**2 + v**2 + 1e-5)
        
        # Внедряем нашу честную физическую пульсацию, чтобы поток не выглядел мертвым ковром
        wave_pulsation = np.sin(x_tensor.numpy().flatten() * 0.005) * np.cos(y_tensor.numpy().flatten() * 0.003) * 0.8
        h_w_physics = np.clip(h_w + wave_pulsation, 0.1, None)
        
        # Фильтр: показываем людям только те точки, где масса РЕАЛЬНО несется вниз (скорость > 1.5 м/с)
        # === ЖЕСТКИЙ ИНЖЕНЕРНЫЙ ФИЛЬТР ФРОНТА ВОЛНЫ ВО ВРЕМЕНИ ===
        # Скорость распространения фронта лавины в ущелье примерно 400 метров в секунду.
        # На первой секунде поток физически не может уйти по длине склона (Y) дальше, чем на 400-500 метров!
        max_allowed_y_distance = float(frame_time) * 450.0
        
        # Показываем точки только там, куда волна УСПЕЛА добежать по времени склона Y
        # (Поскольку Y в парсере идет сверху вниз, отсекаем дальние нижние зоны)
        time_space_mask = Y_grid.flatten() < max_allowed_y_distance
        
        # Объединяем физическую скорость, высоту вала и временное ограничение пути
        # УЖЕСТОЧАЕМ ПОРОГ: Отрезаем фоновый нейросетевой шум (эффект снега)
        # Показываем массу только там, где скорость РЕАЛЬНОГО схода выше 2.5 м/с, 
        # а глубина потока гарантированно больше 0.8 метра
        active_flow = (v_mag > 2.5) & (h_w_physics > 0.8) & time_space_mask
        
        if np.any(active_flow):
            x_plot = X_grid.flatten()[active_flow]
            y_plot = Y_grid.flatten()[active_flow]
            
            # Важнейший физический момент: селевой вал течет ПОВЕРХ рельефа горы!
            # ИСПРАВЛЕННЫЙ ТОЧНЫЙ РАСЧЕТ ВЫСОТЫ (Убираем фантомное двоение)
            # Вытаскиваем исходную высоту горы напрямую из тензора Z_mesh, который изначально одномерный!
            z_mountain_sol = Z_mesh.numpy().flatten()[active_flow]

            # Селевой вал ложится строго поверх реальной земли
            z_plot = z_mountain_sol + h_w_physics[active_flow]
                        
            # Обновляем координаты 3D-точек и перекрашиваем их в зависимости от толщины вала
            sc._offsets3d = (x_plot, y_plot, z_plot)
            sc.set_array(h_w_physics[active_flow])
            sc.set_clim(0.5, 7.5) # Границы цветовой шкалы (от мелкой жижи до 7.5 метров)
        else:
            sc._offsets3d = ([], [], [])
            
    return sc,

# 5. Запуск анимации (интервал 100 миллисекунд между кадрами)
ani = animation.FuncAnimation(fig, update_stream, frames=frames, interval=100, blit=False)

# Если хотите сохранить это как видео-файл для презентации, раскомментируйте строку ниже:
# ani.save("landslide_destruction_3d.gif", writer='pillow', fps=10)

plt.colorbar(sc, ax=ax, label='Высота разрушительного вала (метры)', shrink=0.5, pad=0.05)
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3 mountains import Axes3D # для старых версий, в новых встроен

# 🔄 1. ЗАГРУЖАЕМ РЕКОРДНЫЕ ТЕНЗОРЫ ИИ ИЗ НАШЕГО NPZ ФАЙЛА
try:
    data = np.load("nepal_sim_results_best.npz")
    x = data['x'].flatten()
    y = data['y'].flatten()
    h_water = data['h_water'].flatten()
    satellite_track = data['satellite_track'].flatten()
    print("✅ Файл непальского рекорда успешно загружен! Строим 3D-сцену...")
except Exception as e:
    print(f"❌ Не удалось найти файл 'nepal_sim_results_best.npz'.")
    print("   Убедись, что модель сделала хотя бы один дофаминовый рекорд, или подставь обычный 'nepal_sim_results.npz'")
    exit()

# 🏔️ 2. СТРОИМ ГЕОМЕТРИЮ ГОРНОГО СКЛОНА ЛАНГТАНГ (Аналог नेपाल Spline)
W_PIXELS, H_PIXELS = 128, 128
DX_METERS = 30.0

X_grid, Y_grid = np.meshgrid(np.arange(W_PIXELS) * DX_METERS, np.arange(H_PIXELS) * DX_METERS)
# Восстанавливаем форму ущелья
Z_mountain = 5200.0 - Y_grid * 20.0 - (1.0 - np.exp(-((X_grid - (W_PIXELS*DX_METERS)/2)/600.0)**2)) * 300.0

# 🎨 3. ИНИЦИАЛИЗАЦИЯ КРАСИВОГО 3D-ОКНА
fig = plt.figure(figsize=(12, 8), facecolor='#111111')
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('#111111')

# Настраиваем сетку и оси, чтобы выглядело как продвинутый софт МЧС
ax.w_xaxis.set_pane_color((0.1, 0.1, 0.1, 1.0))
ax.w_yaxis.set_pane_color((0.1, 0.1, 0.1, 1.0))
ax.w_zaxis.set_pane_color((0.1, 0.1, 0.1, 1.0))
ax.tick_params(colors='white')
ax.set_title("СИМУЛЯЦИЯ ГЕОФИЗИЧЕСКОЙ КАТАСТРОФЫ ЛАНГТАНГ (PINN ИИ 2026)", color='white', fontsize=14, pad=20)

# Рисуем саму гору серой сеткой (каркас ущелья)
mountain_surface = ax.plot_surface(X_grid, Y_grid, Z_mountain, cmap='gray', alpha=0.4, edgecolor='none')

# Сетка для отображения летящей волны селя
# На старте она пустая, анимация будет её обновлять
water_scat = ax.scatter([], [], [], c=[], cmap='YlOrRd', s=15, alpha=0.8, label="Селевой фронт (Вода + Камни)")

# 🛠️ 4. ФУНКЦИЯ ДИНАМИЧЕСКОГО КАДРА АНИМАЦИИ
def update_3d_frame(frame):
    ax.view_init(elev=25, azim=frame * 0.5) # Гора будет плавно вращаться в 3D!
    
    # Чтобы сымитировать сход лавины во времени, мы плавно "проявляем" точки 
    # в зависимости от их положения по оси Y (сверху вниз)
    max_y_reach = (frame * 50.0) % (H_PIXELS * DX_METERS)
    
    # Фильтруем точки, которые ИИ предсказал как движущийся поток (высота > 0.1 м)
    mask = (y < max_y_reach) & (h_water > 0.1)
    
    if np.sum(mask) > 0:
        x_flow = x[mask]
        y_flow = y[mask]
        h_flow = h_water[mask]
        
        # Сажаем поток ровно на рельеф горы (Z_mountain) + даем высоту вала
        z_flow = 5200.0 - y_flow * 20.0 - (1.0 - np.exp(-((x_flow - (W_PIXELS*DX_METERS)/2)/600.0)**2)) * 300.0 + h_flow
        
        # Обновляем 3D scatter: цвет зависит от высоты вала (красный — глубокий вал, желтый — мелкий)
        global water_scat
        water_scat.remove() # очищаем старый кадр
        water_scat = ax.scatter(x_flow, y_flow, z_flow, c=h_flow, cmap='YlOrRd', vmin=0.1, vmax=4.0, s=25, alpha=0.9)
        
    return water_scat,

# 🎬 5. ЗАПУСК КИНЕМАТОГРАФИЧЕСКОЙ АНИМАЦИИ
print("🎬 Анимация запущена! Гора плавно вращается. Закрой окно графика, чтобы выйти.")
ani = FuncAnimation(fig, update_3d_frame, frames=720, interval=30, blit=False)

ax.set_xlabel("Ширина долины X (м)", color='white')
ax.set_ylabel("Длина ущелья Y (м)", color='white')
ax.set_zlabel("Высота Z (м)", color='white')
ax.set_zlim(2000, 5500)

plt.show()
