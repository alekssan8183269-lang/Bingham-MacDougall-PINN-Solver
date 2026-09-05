
import numpy as np
import random

class AntColonyTracer:
    def __init__(self, width=128, height=128, num_ants=500, alpha_evaporation=0.1):
        """
        Роевой био-инвертор траектории селевого потока.
        width, height: Размеры DEM-карты рельефа Непала
        num_ants: Количество жестко фиксированных муравьев-частиц (аппаратный запрет на взрыв массы)
        alpha_evaporation: Коэффициент испарения феромонного следа (диссипация старой памяти)
        """
        self.W = width
        self.H = height
        self.num_ants = num_ants
        self.alpha = alpha_evaporation
        
        # Матрица феромонов (карта фокуса для PINN градиентов)
        self.pheromone_map = np.zeros((self.H, self.W), dtype=np.float32)
        
        # Фиксированный рой агентов. Каждый муравей имеет координаты и массу
        self.ants = []
        self.reset_colony()

    def reset_colony(self):
        """Спавн роя на вершине горы Лангтанг Лирунг (Инициация обрушения ледника)"""
        self.ants = []
        for _ in range(self.num_ants):
            # Спавним муравьев в верховьях (верхняя правая зона по координатам статьи Nature)
            init_x = float(random.uniform(self.W * 0.7, self.W * 0.9))
            init_y = float(random.uniform(self.H * 0.7, self.H * 0.9))
            self.ants.append({
                'x': init_x, 
                'y': init_y, 
                'active': True,
                'stuck_frames': 0
            })

    def run_insect_simulation(self, dem_spline, max_steps=150):
        """
        Запуск муравьиного конвейера. Без Navier-Stokes и автограда.
        dem_spline: Готовый RectBivariateSpline вашей карты высот
        """
        # 1. Испаряем старый феромонный след (зачистка шума прошлых эпох)
        self.pheromone_map *= (1.0 - self.alpha)
        
        grid_cell_area = 30.0 * 30.0 # Площадь одной ячейки (DX=30м)
        
        for step in range(max_steps):
            for ant in self.ants:
                if not ant['active']:
                    continue
                
                cx, cy = int(ant['x']), int(ant['y'])
                
                # Защита от вылета за границы матрицы рельефа
                if cx <= 1 or cx >= self.W - 2 or cy <= 1 or cy >= self.H - 2:
                    ant['active'] = False
                    continue
                
                # 2. Локальный гео-рентген: Берем уклоны строго в текущей точке
                # Муравей не перебирает всю карту, он видит только то, что под лапками
                dz_dx = float(dem_spline(ant['x'], ant['y'], dx=1, dy=0, grid=False))
                dz_dy = float(dem_spline(ant['x'], ant['y'], dx=0, dy=1, grid=False))
                
                # 3. Биологический шаг (Ленивое падение по градиенту + броуновский хрип дешевого динамика)
                # Уклон заставляет катиться вниз, а случайный шум симулирует обход валунов
                dx = -dz_dx * 0.1 + random.uniform(-0.8, 0.8)
                dy = -dz_dy * 0.1 + random.uniform(-0.8, 0.8)
                
                step_len = np.hypot(dx, dy)
                if step_len < 0.05:
                    ant['stuck_frames'] += 1
                else:
                    ant['stuck_frames'] = 0
                    
                # Если муравей застрял в локальной яме (депрессионная воронка) — он цементируется
                if ant['stuck_frames'] > 10:
                    ant['active'] = False
                    continue
                    
                # Двигаем агента
                ant['x'] += dx
                ant['y'] += dy
                
                # 4. Сброс феромона: Муравей метит ячейку, подтверждая физическую проходимость русла
                # Сила метки пропорциональна крутизне склона (кинетическому разгону)
                nx, ny = int(ant['x']), int(ant['y'])
                if 0 <= nx < self.W and 0 <= ny < self.H:
                    self.pheromone_map[ny, nx] += 1.0 + float(np.abs(dz_dx) + np.abs(dz_dy)) * 2.0

        # Нормализуем карту фокуса (0.0 - 1.0)
        max_ph = np.max(self.pheromone_map)
        if max_ph > 0:
            self.pheromone_map /= max_ph

    def extract_pinn_training_points(self, threshold=0.1, batch_size=500):
        """
        ГЛАВНЫЙ ЧИТ: Извлекает из матрицы феромонов только 'живые' координаты.
        Вместо 16384 случайных точек ИИ будет обучаться на батче из русла муравьев!
        """
        active_y, active_x = np.where(self.pheromone_map > threshold)
        
        # Если рой никуда не дотек (защита от пустого батча) — спавним дефолтную область
        if len(active_x) < 50:
            return (np.random.rand(batch_size, 1) * self.W * 30.0, 
                    np.random.rand(batch_size, 1) * self.H * 30.0)
            
        sampled_indices = np.random.choice(len(active_x), size=batch_size, replace=True)
        
        # Переводим пиксели обратно в честные метры ущелья Непала
        x_meters = (active_x[sampled_indices].astype(np.float32) * 30.0).reshape(-1, 1)
        y_meters = (active_y[sampled_indices].astype(np.float32) * 30.0).reshape(-1, 1)
        
        return x_meters, y_meters



     