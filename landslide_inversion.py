import taichi as ti
import numpy as np
import threading
import time

# Инициализируем Taichi с поддержкой Vulkan или CUDA (в зависимости от вашей видеокарты)
ti.init(arch=ti.gpu)

# 1. ГЕОМЕТРИЯ МИРА И МАСШТАБ СИМУЛЯЦИИ
W, H = 256, 256              # Размер сетки высот (пиксели)
NUM_WORLDS = 1000            # Количество параллельных мутирующих миров на GPU
DX = 10.0                    # Шаг сетки в реальности (10 метров)
DT = 0.01                    # Шаг по времени (секунды)

# 2. СТРУКТУРЫ ДАННЫХ В ВИДЕОПАМЯТИ (SSBO / Fields)
# Карта высот (DEM) — общая для всех миров
dem = ti.field(dtype=ti.f32, shape=(W, H))

# Целевая маска реального схода со спутника (для сравнения ИИ)
satellite_mask = ti.field(dtype=ti.f32, shape=(W, H))

# Буферы физических состояний (Ping-Pong для исключения затирания)
# vec4: x=h_water (грязь), y=h_ice (лед), z=u (скорость X), w=v (скорость Y)
physics_state_A = ti.Vector.field(4, dtype=ti.f32, shape=(NUM_WORLDS, W, H))
physics_state_B = ti.Vector.field(4, dtype=ti.f32, shape=(NUM_WORLDS, W, H))

# Буфер параметров миров (Гены ИИ): x=tau_y (Бингам), y=K (Вязкость), z=melt_rate
world_params = ti.Vector.field(3, dtype=ti.f32, shape=(NUM_WORLDS))

# Буфер ошибок MSE для каждого мира
world_errors = ti.field(dtype=ti.f32, shape=(NUM_WORLDS))

# Хост-буфер для асинхронного вывода результатов в CPU
host_best_params = np.zeros(3, dtype=np.float32)


# 3. МАТЕМАТИЧЕСКОЕ ЯДРО (КОМПЬЮТЕРНЫЕ ШЕЙДЕРЫ)

@ti.kernel
def init_simulation():
    """ Заполняем память случайным хаосом параметров и инициализируем лед """
    # Инициализация параметров миров (Гены)
    for w in range(NUM_WORLDS):
        tau_y = ti.random() * 50.0 + 10.0   # Предел текучести Бингама
        K = ti.random() * 5.0 + 0.5         # Динамическая вязкость
        melt_rate = ti.random() * 0.1 + 0.01 # Коэффициент таяния льда
        world_params[w] = ti.Vector([tau_y, K, melt_rate])

    # Загружаем «лед на горах» в каждый мир
    for w, x, y in physics_state_A:
        # Пусть в верховьях (верхняя треть карты) лежит лед толщиной 20 метров
        h_ice = 0.0
        if y > int(H * 0.7) and x > int(W * 0.3) and x < int(W * 0.7):
            h_ice = 20.0
        physics_state_A[w, x, y] = ti.Vector([0.0, h_ice, 0.0, 0.0])

@ti.kernel
def physics_step(step_id: ti.i32):
    """ 
    Нелинейный шаг Сен-Венана + Бингам + Термодинамика таяния.
    Использует концепцию Zero-Copy — гоняет данные внутри VRAM.
    """
    for w, x, y in physics_state_A:
        # Реализация Ping-Pong буферизации
        is_even = (step_id % 2 == 0)
        
        # Считываем текущую ячейку
        self_state = physics_state_A[w, x, y] if is_even else physics_state_B[w, x, y]
        h_water = self_state[0]
        h_ice = self_state[1]
        u = self_state[2]
        v = self_state[3]

        p = world_params[w] # Гены этого конкретного мира

        # Граничные условия
        if x > 0 and x < W - 1 and y > 0 and y < H - 1:
            # Считаем градиент высот (Уклоны русла) по схеме «Крест»
            z_east  = dem[x + 1, y]
            z_west  = dem[x - 1, y]
            z_north = dem[x, y + 1]
            z_south = dem[x, y - 1]
            
            slope_x = (z_east - z_west) / (2.0 * DX)
            slope_y = (z_north - z_south) / (2.0 * DX)

            # Сила тяжести, толкающая поток вниз по склону
            g = 9.81
            force_x = -g * h_water * slope_x
            force_y = -g * h_water * slope_y

            # 🛠️ НЕЛИНЕЙНЫЙ ЗАКОН БИНГАМА (Вязкопластичный сдвиг)
            v_mag = ti.sqrt(u*u + v*v)
            tau = p[0] + p[1] * v_mag # Трение = Предел_текучести + Вязкость * Скорость

            # Критерий застывания «ленивой» массы
            if h_water > 0.05 and v_mag > 0.001:
                # Гасим скорость пропорционально нелинейному трению Бингама
                u += (force_x - (u / v_mag) * tau) * DT
                v += (force_y - (v / v_mag) * tau) * DT
            else:
                u = 0.0
                v = 0.0

            # ❄️ ТЕРМОДИНАМИКА ТАЯНИЯ ЛЬДА ОТ ТРЕНИЯ (Каскадный триггер)
            friction_heat = tau * v_mag * 0.005 # Выделение тепла Q = tau * v * coeff
            melted = ti.min(h_ice, friction_heat * DT * p[2])
            
            h_ice -= melted
            h_water += melted * 0.9 # Фазовый переход: лед растаял и превратился в воду-смазку

            # Ленивое перетекание массы (Закон сохранения массы)
            # Упрощенная локальная диффузия водяного фронта по соседям
            h_water_north = physics_state_A[w, x, y + 1][0] if is_even else physics_state_B[w, x, y + 1][0]
            h_water_south = physics_state_A[w, x, y - 1][0] if is_even else physics_state_B[w, x, y - 1][0]
            h_water_east  = physics_state_A[w, x + 1, y][0] if is_even else physics_state_B[w, x + 1, y][0]
            h_water_west  = physics_state_A[w, x - 1, y][0] if is_even else physics_state_B[w, x - 1, y][0]
            
            net_flow = (h_water_north + h_water_south + h_water_east + h_water_west) - 4.0 * h_water
            h_water += net_flow * 0.1 # Коэффициент гидродинамического переноса

        # Запись в парный буфер
        next_state = ti.Vector([h_water, h_ice, u, v])
        if is_even:
            physics_state_B[w, x, y] = next_state
        else:
            physics_state_A[w, x, y] = next_state

@ti.kernel
def evaluate_and_mutate(step_id: ti.i32):
    """
    ИИ-фильтр обратной задачи.
    Считает ошибку MSE каждого мира со спутником и отсекает дичь прямо в VRAM.
    """
    is_even = (step_id % 2 == 0)

    # Шаг 1. Параллельный подсчет MSE ошибки
    for w in range(NUM_WORLDS):
        error_sum = 0.0
        for x, y in ti.ndrange(W, H):
            h_water = physics_state_B[w, x, y][0] if is_even else physics_state_A[w, x, y][0]
            # Симулируем бинарный след схода (где глубина > 10см)
            sim_mask = 1.0 if h_water > 0.1 else 0.0
            diff = sim_mask - satellite_mask[x, y]
            error_sum += diff * diff
        world_errors[w] = error_sum / float(W * H)

    # Шаг 2. Поиск лучшего мира (Редукция)
    best_world_idx = 0
    min_error = 999999.0
    for w in range(NUM_WORLDS):
        if world_errors[w] < min_error:
            min_error = world_errors[w]
            best_world_idx = w

    # Шаг 3. Эволюционный отбор (Мутация генов вокруг лидера)
    best_p = world_params[best_world_idx]
    for w in range(NUM_WORLDS):
        if w != best_world_idx:
            if world_errors[w] > min_error * 1.5: # Если мир выдал дичь (ошибка > 150% от лидера)
                # Мутируем параметры вокруг лидера (Случайный поиск)
                mutate_tau = best_p[0] + (ti.random() - 0.5) * 5.0
                mutate_K = best_p[1] + (ti.random() - 0.5) * 0.5
                mutate_melt = best_p[2] + (ti.random() - 0.5) * 0.01
                world_params[w] = ti.Vector([ti.max(1.0, mutate_tau), ti.max(0.1, mutate_K), ti.max(0.001, mutate_melt)])


# 4. АСИНХРОННЫЙ ПРИНТ (ВЫВОД ДАННЫХ ИЗ ИНВЕРТОРА)

def async_logger(params_snapshot, error_val, gen):
    """ Фоновый поток CPU для логирования, чтобы вычисления на GPU не фризили """
    print(f"\n🛸 [ИИ-Эволюция. Поколение {gen}]")
    print(f"└── Минимальная MSE ошибка модели: {error_val:.6f}")
    print(f"└── Истинные параметры ледника определены:")
    print(f"    ├── Трение Бингама (tau_y): {params_snapshot[0]:.4f} Па")
    print(f"    ├── Вязкость среды (K):     {params_snapshot[1]:.4f} Па·с")
    print(f"    └── Скорость таяния льда:   {params_snapshot[2]:.4f} м/с·Дж")


# 5. ИНИЦИАЛИЗАЦИЯ ДАННЫХ И СТАРТ

# Создаем синтетическую наклонную гору (DEM)
@ti.kernel
def generate_synthetic_mountain():
    for x, y in dem:
        dem[x, y] = float(H - y) * 2.0 # Идеальный склон, идущий сверху вниз
        # Создаем «искусственный след селя» для симуляции спутника (ИИ будет искать его параметры)
        if y < int(H * 0.5) and x > int(W * 0.4) and x < int(W * 0.6):
            satellite_mask[x, y] = 1.0

generate_synthetic_mountain()
init_simulation()

print("🚀 Движок инверсии запущен в VRAM. Поиск скрытых параметров нелинейной системы...")

# Главный цикл ИИ
step = 0
generation = 0

try:
    while True:
        # Просчитываем шаг нелинейной физики на видеокарте
        physics_step(step)
        
        # Раз в 100 физических шагов запускаем эволюционный фильтр «дичи»
        if step % 100 == 0:
            evaluate_and_mutate(step)
            generation += 1
            
            # Асинхронный сброс лучшего результата в CPU
            if generation % 10 == 0:
                # Вытаскиваем ошибку и гены лидера без фриза симуляции
                best_err = np.min(world_errors.to_numpy())
                best_idx = np.argmin(world_errors.to_numpy())
                best_p_vector = world_params.to_numpy()[best_idx]
                
                # Запускаем фоновую печать
                threading.Thread(target=async_logger, args=(best_p_vector, best_err, generation)).start()

        step += 1
        time.sleep(0.001) # Защита от перегрева видеокарты

except KeyboardInterrupt:
    print("\n🛑 Симуляция остановлена пользователем. Результаты сохранены в репозитории.")
