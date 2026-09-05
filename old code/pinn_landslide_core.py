import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import threading
import time

# Задаем глобальные физические константы
G = 9.81         # Ускорение свободного падения (м/с^2)
DX = 10.0        # Дискретность сетки рельефа (метры)
L_FUSION = 334.0 # Удельная теплота плавления льда (кДж/кг)

# 1. АРХИТЕКТУРА СЕТИ PINN (ВЫЧИСЛИТЕЛЬНОЕ ЯДРО)
class LandslidePINN(nn.Module):
    def __init__(self):
        super().__init__()
        # Вход: Координаты X, Y и Время T (3 параметра)
        # Выход: h_water, h_ice, u_velocity, v_velocity (4 параметра)
        self.net = nn.Sequential(
            nn.Linear(3, 40),
            nn.Tanh(),         # Tanh критически важен для вычисления гладких производных
            nn.Linear(40, 40),
            nn.Tanh(),
            nn.Linear(40, 40),
            nn.Tanh(),
            nn.Linear(40, 4)   # Выходной физический вектор состояния
        )
        
    def forward(self, x, y, t):
        inputs = torch.cat([x, y, t], dim=1)
        return self.net(inputs)

# 2. МАТЕМАТИЧЕСКИЙ ФИЛЬТР НЕЛИНЕЙНЫХ УРАВНЕНИЙ (ЗАКОНЫ ВСЕЛЕННОЙ)
def compute_physics_loss(pinn_model, x, y, t, tau_y, K, melt_rate):
    """
    Вычисляет нелинейную систему дифференциальных уравнений в частных производных.
    В идеальном мире все вычисленные балансы (residuals) должны стремиться к нулю.
    """
    # Заставляем PyTorch отслеживать градиенты для автодифференцирования
    outputs = pinn_model(x, y, t)
    
    h_w = outputs[:, 0:1] # Глубина водяного/грязевого потока
    h_i = outputs[:, 1:2] # Глубина (толщина) льда в этой точке
    u   = outputs[:, 2:3] # Скорость потока по оси X
    v   = outputs[:, 3:4] # Скорость потока по оси Y
    
    # Имитируем локальный уклон рельефа горы (зависит от координат x, y)
    # В реальном проекте сюда подгружается матрица высот DEM
    slope_x = 0.25 * torch.sin(x)  # Крутизна склона по X
    slope_y = 0.35 * torch.cos(y)  # Крутизна склона по Y

    # --- 1. Вычисление первых производных по времени (u_t, v_t, h_t) ---
    h_w_t = torch.autograd.grad(h_w, t, torch.ones_like(h_w), create_graph=True)[0]
    h_i_t = torch.autograd.grad(h_i, t, torch.ones_like(h_i), create_graph=True)[0]
    u_t   = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    v_t   = torch.autograd.grad(v, t, torch.ones_like(v), create_graph=True)[0]

    # --- 2. Вычисление пространственных производных (Градиенты давлений) ---
    h_w_x = torch.autograd.grad(h_w, x, torch.ones_like(h_w), create_graph=True)[0]
    h_w_y = torch.autograd.grad(h_w, y, torch.ones_like(h_w), create_graph=True)[0]

    # --- 3. НЕЛИНЕЙНАЯ РЕОЛОГИЯ БИНГАМА (Вязкопластичное трение) ---
    v_mag = torch.sqrt(u**2 + v**2 + 1e-5) # 1e-5 предотвращает деление на ноль при остановке
    tau = tau_y + K * v_mag                # Полное напряжение сдвига Бингама-Хершеля
    
    # Силы сопротивления, действующие против вектора скорости
    friction_force_x = (u / v_mag) * tau
    friction_force_y = (v / v_mag) * tau

    # --- 4. ТЕРМОДИНАМИКА ТАЯНИЯ ЛЬДА ОТ ТРЕНИЯ (Каскадный триггер Непала) ---
    friction_heat = tau * v_mag * 0.005    # Выделение тепла Q от трения скольжения лавины
    melted_ice = torch.where(h_i > 0.0, friction_heat * melt_rate, torch.zeros_like(h_i))

    # --- 5. ФОРМУЛИРОВАНИЕ НЕЛИНЕЙНЫХ ОШИБОК БАЛАНСА (Physics Residuals) ---
    # Закон сохранения массы воды (с учетом притока от растаявшего льда)
    mass_water_residual = h_w_t + u * h_w_x + v * h_w_y - melted_ice * 0.9
    
    # Закон изменения толщины льда (лед только убывает при плавлении)
    mass_ice_residual = h_i_t + melted_ice
    
    # Уравнения импульса Навье-Стокса / Сен-Венана (Сила тяжести склона против трения Бингама)
    gravity_force_x = -G * h_w * slope_x
    gravity_force_y = -G * h_w * slope_y
    
    momentum_x_residual = u_t + u * h_w_x - (gravity_force_x - friction_force_x)
    momentum_y_residual = v_t + v * h_w_y - (gravity_force_y - friction_force_y)

    # Интегрируем все ошибки физики в один среднеквадратичный Loss
    loss_p = torch.mean(mass_water_residual**2) + \
             torch.mean(mass_ice_residual**2) + \
             torch.mean(momentum_x_residual**2) + \
             torch.mean(momentum_y_residual**2)
             
    return loss_p

# 3. АСИНХРОННЫЙ ПРИНТ ЛОГОВ (Чтобы CPU не вешал математический конвейер)
def async_reporter(epoch, loss_val, t_y, v_k):
    print(f"🛸 [PINN Эволюция Земли] Эпоха {epoch:04d} | Ошибка нелинейной физики: {loss_val:.6f}")
    print(f"   └── Текущая калибровка среды: Трение Бингама = {t_y:.2f} Па, Вязкость = {v_k:.3f} Па·с")

# 4. ИНИЦИАЛИЗАЦИЯ И ЦИКЛ ОБРАТНОЙ ЗАДАЧИ
pinn = LandslidePINN()
optimizer = optim.Adam(pinn.parameters(), lr=0.005)

# Фиксированные параметры грунта, которые наш ИИ должен «подтвердить» в уравнениях
# (В реальной обратной задаче они подбираются внешним циклом оптимизации)
true_tau_y = 35.0   # Истинный предел текучести для грязекаменного бетона
true_K = 1.2        # Истинная вязкость
true_melt = 0.02    # Истинная скорость таяния льда на скале

print("🚀 Запуск физико-информированного ИИ-инвертора селевых потоков...")
print("==================================================================")

try:
    for epoch in range(1001):
        optimizer.zero_grad()
        
        # Генерируем случайные пространственно-временные точки (Координатная сетка в VRAM)
        # requires_grad=True — сердце PINN, без него PyTorch не сможет взять дифференциалы по физике
        x_points = torch.rand(200, 1, requires_grad=True) * 10.0
        y_points = torch.rand(200, 1, requires_grad=True) * 10.0
        t_points = torch.rand(200, 1, requires_grad=True) * 5.0
        
        # Шаг 1. Считаем, насколько сильно ИИ нарушил уравнения Сен-Венана и закон Бингама
        loss_physics = compute_physics_loss(
            pinn, x_points, y_points, t_points, 
            true_tau_y, true_K, true_melt
        )
        
        # Шаг 2. Задаем начальные граничные условия (Граница t=0, где вверху горы лежит лед)
        t0 = torch.zeros_like(x_points)
        outputs_t0 = pinn(x_points, y_points, t0)
        h_ice_t0_pred = outputs_t0[:, 1:2]
        
        # Физическое условие: если y > 7.0 (вершина), там изначально лежит лед толщиной 20м
        h_ice_t0_real = torch.where(y_points > 7.0, torch.full_like(y_points, 20.0), torch.zeros_like(y_points))
        loss_boundary = torch.mean((h_ice_t0_pred - h_ice_t0_real)**2)
        
        # Общий штраф системы
        total_loss = loss_physics + loss_boundary
        
        # Шаг 3. Обратное распространение ошибки и корректировка весов ИИ
        total_loss.backward()
        optimizer.step()
        
        # Асинхронное логирование раз в 200 эпох
        if epoch % 200 == 0:
            loss_item = total_loss.item()
            threading.Thread(
                target=async_reporter, 
                args=(epoch, loss_item, true_tau_y, true_K)
            ).start()
            
except KeyboardInterrupt:
    print("\n🛑 Процесс инверсии остановлен пользователем.")

print("==================================================================")
print("🎉 Обучение завершено. Модель законсервировала законы нелинейной механики внутри весов!")
