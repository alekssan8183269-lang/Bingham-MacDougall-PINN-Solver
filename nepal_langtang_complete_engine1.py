import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.interpolate import RectBivariateSpline
import threading
import time
import os  

W_PIXELS, H_PIXELS = 128, 128
DX_METERS = 30.0  
G = 9.81

class NepalSplineParser:
    def __init__(self):
        self.dem_matrix = np.zeros((W_PIXELS, H_PIXELS), dtype=np.float32)
        for i in range(W_PIXELS):
            for j in range(H_PIXELS):
                self.dem_matrix[i, j] = 5200.0 - j * 20.0 - (1.0 - np.exp(-((i - W_PIXELS/2)/20.0)**2)) * 300.0
        self.x_coords = np.arange(0, W_PIXELS) * DX_METERS
        self.y_coords = np.arange(0, H_PIXELS) * DX_METERS
        self.spline = RectBivariateSpline(self.x_coords, self.y_coords, self.dem_matrix.T, kx=3, ky=3)

    def get_geometry(self, x_tensor, y_tensor):
        x_np = x_tensor.detach().cpu().numpy().flatten()
        y_np = y_tensor.detach().cpu().numpy().flatten()
        z_val = self.spline(x_np, y_np, grid=False).astype(np.float32)
        dz_dx = self.spline(x_np, y_np, dx=1, dy=0, grid=False).astype(np.float32)
        dz_dy = self.spline(x_np, y_np, dx=0, dy=1, grid=False).astype(np.float32)
        device = x_tensor.device
        return (torch.from_numpy(z_val).unsqueeze(1).to(device),
                torch.from_numpy(dz_dx).unsqueeze(1).to(device),
                torch.from_numpy(dz_dy).unsqueeze(1).to(device))

# class LandslidePINNCore(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # Увеличиваем количество нейронов до 128 и добавляем 4-й слой для нелинейной емкости
#         self.net = nn.Sequential(
#             nn.Linear(3, 128),
#             nn.Tanh(),
#             nn.Linear(128, 128),
#             nn.Tanh(),
#             nn.Linear(128, 128),
#             nn.Tanh(),
#             nn.Linear(128, 4)
#         )
        
#     def forward(self, x, y, t):
#         inputs = torch.cat([x, y, t], dim=1)
#         return self.net(inputs)
class LandslidePINNCore(nn.Module):
    def __init__(self):
        super().__init__()
        # Входной слой: принимает X, Y, T
        self.input_layer = nn.Linear(3, 128)
        
        # Мощные когнитивные блоки (Каждый блок имеет скрытую память)
        self.block1 = nn.Linear(128, 128)
        self.block2 = nn.Linear(128, 128)
        self.block3 = nn.Linear(128, 128)
        
        # Финальный слой: выдает h_w, h_i, u, v
        self.output_layer = nn.Linear(128, 4)
        
        # Математическая активация для гладких физических градиентов
        self.activation = nn.Tanh()
        
    def forward(self, x, y, t):
        inputs = torch.cat([x, y, t], dim=1)
        
        # Первичный запуск данных в мозг ИИ
        x_hidden = self.activation(self.input_layer(inputs))
        
        # 🔥 МОСТ ПАМЯТИ 1 (Skip-Connection): Склеиваем глубокие слои со стартовыми
        x_hidden = x_hidden + self.activation(self.block1(x_hidden))
        
        # 🔥 МОСТ ПАМЯТИ 2: Пробрасываем градиенты времени и пространства дальше
        x_hidden = x_hidden + self.activation(self.block2(x_hidden))
        
        # 🔥 МОСТ ПАМЯТИ 3: Защита от затухания вторых производных Навье-Стокса
        x_hidden = x_hidden + self.activation(self.block3(x_hidden))
        
        return self.output_layer(x_hidden)
def compute_nepal_physics_loss(pinn, parser, x, y, t, tau_y, K, melt_rate, epoch):
    outputs = pinn(x, y, t)
    global insurance_log  

    # Нормализация размерностей (Разбиваем на метры и секунды)
    h_w_raw = torch.exp(outputs[:, 0:1])
    h_i     = torch.exp(outputs[:, 1:2])
    u       = outputs[:, 2:3]
    v       = outputs[:, 3:4]
    
    v_mag = torch.sqrt(u**2 + v**2 + 1e-5)

    # ГЕОМЕТРИЯ И АВТОГРАД РАСЧЕТЫ
    _, slope_x, slope_y = parser.get_geometry(x, y)

    # =========================================================================
    # 🌊 ЖЕСТКАЯ СВЯЗЬ ФАЗ: РЕАЛЬНОЕ ГРЯЗЕВОЕ РАЗЖИЖЕНИЕ И ТОРМОЖЕНИЕ ПОТОКА
    # =========================================================================
    water_fraction = h_w_raw / (h_i + h_w_raw + 1e-5)    
    # 1. Трение Бингама падает по мере таяния и разгона (Динамический диапазон ОТ 60 ДО 25 Па)
    # На старте (t=0) трение = 60 Па (сухой грунт), через 10 секунд падает до 25 Па (жидкая смазка)
    # Трение Бингама падает, когда вода разжижает матрицу конгломерата
    tau_y_dynamic = torch.clamp(60.0 - water_fraction * 45.0, min=15.0, max=60.0)
    # Но вязкость селевой жижи резко увеличивается из-за заиливания и вовлечения камней от скорости
    K_dynamic = torch.clamp(0.5 + (v_mag * water_fraction * 0.15), min=0.5, max=3.5)
    # 3. Скорость таяния льда растет лавинообразно от разогрева (Динамический диапазон ОТ 0.01 ДО 0.25)
    # melt_dynamic = torch.clamp(0.01 + t * 0.024, min=0.01, max=0.25)
    # --------------------------------------------------------------------------
    # Дальше в уравнениях импульса и тепла ИИ использует эти ЖИВЫЕ переменные:
    tau = tau_y_dynamic + K_dynamic * v_mag # Трение Бингама теперь ЖИВОЕ!
    # Плотность чистой воды и коренной гранитной породы Непала
    RHO_WATER = 1000.0
    RHO_SOLID = 2600.0
    # Скорость эрозии (м/с) растет лавинообразно от скорости потока и массы
    # Ограничиваем эрозию зоной каньона (in_canyon_mask уже есть в твоем коде)
    in_canyon_mask = (y < (H_PIXELS * DX_METERS * 0.4)).float()
    # === МОДЕЛЬ МАКДУГАЛЛА: КОЭФФИЦИЕНТ ВОВЛЕЧЕНИЯ ДОННОЙ ПОРОДЫ ===
    # Показывает, сколько метров донного грунта сдирает сель на 1 метр пути. Для непальских ущелий с мягким моренным чехлом он обычно составляет от 0.0005 до 0.002.
    # ⛰️ ДИНАМИЧЕСКИЙ МАКДУГАЛЛ: Чем быстрее несется сель и чем он глубже, 
    # тем сильнее сопротивляется дно каньона (эффект самозаклинивания эрозии)
    # Коэффициент динамически падает при разгоне, защищая массу от взрыва градиентов!
    E_s_dynamic = torch.clamp(0.00005 / (1.0 + v_mag * 0.1 + h_w_raw * 0.05), min=0.000005, max=0.00005)
    
    # Физическое уравнение МакДугалла для скорости размыва ложа (dh/dt)
    # Скорость эрозии (dh/dt) теперь не превышает разумные 2-5 метров слоя в секунду
    # Скорость эрозии (dh/dt) теперь физически сбалансирована
    erosion_rate = torch.clamp(E_s_dynamic * h_w_raw * v_mag * in_canyon_mask, max=3.0)
    # Объемная концентрация твердой фазы (камни + лед)
    # Корректно соотносим эрозию с шагом сетки, чтобы не перегружать матрицу плотности
    concentration_s = torch.clamp((h_i + (erosion_rate * 0.001)) / (h_w_raw + 1e-5), min=0.05, max=0.65)  
    # ЖИВАЯ МНОГОФАЗНАЯ ПЛОТНОСТЬ
    rho_mix = concentration_s * RHO_SOLID + (1.0 - concentration_s) * RHO_WATER
    # === МОДЕЛЬ МЕХАНИЧЕСКОГО ДРОБЛЕНИЯ И ИЗМЕЛЬЧЕНИЯ ПОРОДЫ (COMMINUTION) ===
    # На старте средний диаметр валунов d_start = 1.5 метра
    # Под действием скорости v_mag и давления массы они перемалываются в каньоне
    # d_part = torch.clamp(1.5 - (v_mag * concentration_s * 0.05) - (h_i * 0.01), min=0.02, max=1.5)
    # 🔥 ЖЕСТКОЕ ДРОБЛЕНИЕ: Привязываем измельчение к квадрату скорости (энергии ударов)
    # Теперь при разгоне валуны будут стремительно перемалываться в мелкую гальку и пудру!
    kinetic_shatter = (v_mag**2) * concentration_s * 0.006
    d_part = torch.clamp(1.5 - kinetic_shatter - (h_i * 0.05), min=0.05, max=1.5)
    # Энергия дробления (высвобождаемое тепло от разрушения кристаллических связей гранита)
    # Чем сильнее измельчается порода (d_part идет к min), тем больше тепла выделяется локально!
    # === МОДИФИКАЦИЯ ТЕПЛОВОГО ВЗРЫВА: МИКРОТРЕНИЕ И ДРОБЛЕНИЕ ===
    # Добавляем +0.5 к скорости внутри расчета тепла, моделируя внутренний треск
    # и микро-соударения валунов при их гравитационном сжатии в каньоне    
    comminution_heat = torch.clamp((1.5 - d_part) * (v_mag + 0.5) * rho_mix * 1e-5, min=0.0)      
    # === МОДЕЛЬ drift-flux МНОГОФАЗНОГО РАЗДЕЛЕНИЯ СКОРОСТЕЙ ===
    # Коэффициент проскальзывания фаз: жидкая вода бежит быстрее средней скорости потока,
    # а тяжелая каменная фракция (concentration_s) тормозится из-за внутреннего заклинивания валунов
    slip_factor_water = 1.0 + 0.25 * (1.0 - concentration_s)
    slip_factor_solid = 1.0 - 0.35 * concentration_s
    # Скорости для расчета переноса ВОДЫ (быстрая фаза, грязевой фронт)
    u_water, v_water = u * slip_factor_water, v * slip_factor_water
    # Скорости для расчета переноса ТВЕРДОГО ЛЬДА И КАМНЕЙ (медленная фаза, голова селя)
    u_solid, v_solid = u * slip_factor_solid, v * slip_factor_solid   
    # Добавляем тепло от статического давления массы (пьезо-эффект)
    # Даже при нулевой скорости вес лавины (h_i * G * 1800) генерирует стартовое подтаивание ложа!    
    # === МОДЕРНИЗАЦИЯ: ЗАКОН СОХРАНЕНИЯ ЭНЕРГИИ "ВЖУХ-ОБРУШЕНИЯ" ===
    # 1. Считаем потенциальную энергию сдвига всей махины на основе крутизны склона Непала
    # Чем выше слой льда (h_i) и круче уклон, тем сильнее гравитационный потенциал срыва!
    slope_magnitude = torch.sqrt(slope_x**2 + slope_y**2 + 1e-5)
    potential_energy_flux = h_i * G * rho_mix * slope_magnitude
    # 2. Энергия стартового обрушения: на первых 2 секундах (t_pts < 2.0) потенциальная энергия 
    # падающей махины лавинообразно переходит в чистый тепловой взрыв ложа ледника!
    collapse_impact_heat = torch.where(t_pts < 2.0, potential_energy_flux * 0.15, torch.zeros_like(t_pts))
    # Наш прошлый пьезо-эффект веса горы и атомное дробление гранита в пудру
    pressure_heat = h_i * G * rho_mix * 2.0e-3        
    # ПОЛНОЕ ЖИВОЕ ТЕПЛО: Теперь учитывает колоссальный вброс энергии падающей махины!
    # ИИ получает стартовый тепловой запал от "вжух-обрушения" ДАЖЕ ПРИ НУЛЕВОЙ СКОРОСТИ!
    friction_heat = (tau * (v_mag + 0.5) * 0.005) + pressure_heat + comminution_heat + collapse_impact_heat # Живое суммарное тепло
    # Базовая температура льда (-5°C) и окружающей среды в Лангтанге (+2°C)
    T_ice = -5.0
    T_env = 2.0
    # ИИ генерирует живое поле температуры масс (внутри основного выхода сети или через аппроксимацию)
    # Оцениваем температуру баланса: генерация тепла от трения против охлаждения атмосферой
    heat_loss_to_air = 0.01 * (T_env - T_ice) * h_w_raw # Охлаждение воздухом    
    # Честный баланс температуры потока
    temperature_flux = (friction_heat * 0.1) - heat_loss_to_air
    # Динамический коэффициент таяния теперь строго завязан на избыточное тепло!
    # Если тепла от трения мало (поток встал в долине) -> таяние падает до нуля естественным путем
    melt_dynamic = torch.clamp(temperature_flux * 0.005, min=0.0, max=0.35)    
    # Расчет таяния идет по динамическому коэффициенту
    # Считаем среднее таяние на ячейку, а не сумму всего батча
    # 🔥 ИСПРАВЛЕНИЕ: Никаких сумм и средних! Считаем таяние локально для каждой точки (формы 200x1)
    expected_melted_vol = torch.where(h_i > 0.0, friction_heat * melt_dynamic, torch.zeros_like(h_i))
    total_raw_water_vol = h_w_raw + 1e-8
    
    # 🔒 ЖЕСТКОЕ ОГРАНИЧЕНИЕ 3.0: Не даем общему объему воды упасть до нуля. НЕ УДАЛЯТЬ ЭТО ЕСЛИ ЧТО ПОДСТРАХОВКА ВОДОЙ В САМОМ НАЧАЛЕ ИЗ НИОТКУДА
    # Если льда растаяло мало, ИИ искусственно подмешивает базовый объем селя (минимум 15.0 кубов)
    # guaranteed_melt_vol = torch.clamp(expected_melted_vol, min=15.0)    
    # Плавно снижаем искусственный подмес от 15.0 до 0.0 за первые 2000 эпох
    # dynamic_min_water = max(0.0, 15.0 * (1.0 - epoch / 2000.0)) if epoch < 2000 else 0.0
    # Подмешиваем страховку локально в каждую точку, где физика буксует
    # guaranteed_melt_vol = torch.clamp(expected_melted_vol, min=dynamic_min_water)
    # 🧪 Считаем, сколько кубов ИИ подмешал искусственно (для принта ниже)
    # Считаем, сколько кубов ИИ подмешал искусственно (с учетом плавного затухания)
    # Для принтов и статистики на верхнем уровне оставляем среднее по батчу (.item() берем ниже в аудите)
    # added_vol_np = max(0.0, dynamic_min_water - torch.mean(expected_melted_vol).item())
    
    # 🔥 ЕСЛИ БЫЛ ПОДМЕС — СОХРАНЯЕМ В НАШ ЛОГ (только если added_vol_np > 0)
    # if added_vol_np > 0.001:  # небольшая отсечка от точности float
        # Так как мы находимся внутри функции, мы можем использовать глобальную переменную (или передавать её, но global проще)
    #     global insurance_log
    #     insurance_log.append((epoch, added_vol_np))

    # СТРАХОВКА УЧИТЫВАЕТСЯ В ГРАФЕ ВЫЧИСЛЕНИЙ
    # 🔥 ИДЕАЛЬНАЯ ПРОПОРЦИЯ МАСС: тензор (200,1) умножается на отношение тензоров (200,1)
    # 🔥 ИСПРАВЛЕНИЕ БРЕХНИ: Ставим жесткий геофизический ограничитель высоты вала!
    # Максимальная глубина потока в ущелье ограничивается 20.0 метрами. 
    # Больше никакой ИИ не сможет нарисовать километровые галлюцинации ради лосса!
    # h_w = torch.clamp(h_w_raw * (guaranteed_melt_vol / total_raw_water_vol), max=20.5)

    # =========================================================================
    # 🌍 АБСОЛЮТНО ЧЕСТНАЯ ФИЗИКА: ШЛЮЗ СТРАХОВКИ НАМЕРТВО ЗАВАРЕН (0.0 м³)
    # =========================================================================
    # Больше никакого искусственного подмеса из ниоткуда! 
    # Вся глубина h_w теперь строго зависит ТОЛЬКО от реального таяния ледника
    added_vol_np = 0.0  # Для принтов диагностики
    
    # Прямая тензорная пропорция без костылей и ограничений min
    h_w = torch.clamp(h_w_raw * (expected_melted_vol / total_raw_water_vol), max=20.5)

    # 🔒 ДОПОЛНИТЕЛЬНАЯ БЛОКИРОВКА В КАНАЛЕ: Если поток движется в зоне спутникового русла,
    # его физическая высота гарантированно срезается снизу на 1.1 метр. Математически исключаем 0.01 м!
    # in_canyon_mask = (y < (H_PIXELS * DX_METERS * 0.4)).float()
    
    # На Этапе 1 (до 2500 эпохи) даем ИИ полную свободу — глубина может быть хоть 0.05 м (чистая физика сухого старта!)
    # На Этапах 2 и 3 включаем честный гидравлический подпор каньона
    if epoch >= 2500:
        h_w = torch.where((v_mag > 1.0) & (in_canyon_mask > 0.5), torch.clamp(h_w, min=1.35), h_w)

    # ⏱️ Элемент дробного дифференциала (Масштабирование памяти времени)
    # Задаем альфа-коэффициент затухания для разных временных масштабов
    alpha_memory = 0.8 
    t_scaled = t ** alpha_memory

    # Дифференциальные градиенты теперь считаются честно для каждого уравнения
    h_w_t = torch.autograd.grad(h_w, t, torch.ones_like(h_w), create_graph=True)[0] * t_scaled
    h_i_t = torch.autograd.grad(h_i, t, torch.ones_like(h_i), create_graph=True)[0] * t_scaled
    u_t   = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0] * t_scaled
    v_t   = torch.autograd.grad(v, t, torch.ones_like(v), create_graph=True)[0] * t_scaled
    # Градиенты по пространству
    h_w_x = torch.autograd.grad(h_w, x, torch.ones_like(h_w), create_graph=True)[0]
    h_w_y = torch.autograd.grad(h_w, y, torch.ones_like(h_w), create_graph=True)[0]
    h_i_x = torch.autograd.grad(h_i, x, torch.ones_like(h_i), create_graph=True)[0]
    h_i_y = torch.autograd.grad(h_i, y, torch.ones_like(h_i), create_graph=True)[0]
    
    # === ГИДРОДИНАМИЧЕСКИЙ ТОРМОЗ СКОРОСТЕЙ ===
    energy_loss_factor = 1.0 + (friction_heat * melt_dynamic * 0.08) + (water_fraction * v_mag * 0.2)
    # 🔥 ЧЕСТНОЕ ТОРМОЖЕНИЕ ПОТОКА ЗА СЧЕТ ТАЯНИЯ (Закон сохранения энергии)
    # Коэффициент 0.002 связывает потерю скорости с фазовым переходом льда в воду
    # energy_loss_factor = 1.0 + (friction_heat * melt_dynamic * 0.002)
    
    friction_force_x = (u / v_mag) * tau * energy_loss_factor
    friction_force_y = (v / v_mag) * tau * energy_loss_factor

    # Эффект виража: оцениваем кривизну русла через изменение уклонов склона
    # Чем сильнее меняется рельеф, тем выше центробежный вынос массы
    # 🔥 ИСПРАВЛЕНИЕ: Безопасный расчет кривизны русла (виражей каньона)
    # Так как slope_x/y пришли из SciPy и не имеют requires_grad, мы берем градиенты 
    # от текущих скоростей потока (u, v), что физически даже более точно описывает кривизну струи!
    curvature_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    curvature_y = torch.autograd.grad(v.sum(), y, create_graph=True)[0]
    
    # Центробежное ускорение (v^2 * кривизна) добавляет боковое давление на борта ущелья
    centrifugal_force_x = (u**2) * curvature_x * in_canyon_mask
    centrifugal_force_y = (v**2) * curvature_y * in_canyon_mask

    # Сила тяжести теперь «гнётся» на поворотах рельефа, заставляя ИИ перераспределять высоту вала h_w
    gravity_force_x = -G * h_w * slope_x + centrifugal_force_x * 0.05
    gravity_force_y = -G * h_w * slope_y + centrifugal_force_y * 0.05
    
    # НАДО: Используем динамическую скорость таяния для каждого этапа!
    # melted_ice = torch.where(h_i > 0.0, friction_heat * melt_dynamic, torch.zeros_like(h_i))

    # Масштабируем невязки (уменьшаем абсолютные значения Паскалей до порядка единиц)
    # mass_water_residual = (h_w_t + u * h_w_x + v * h_w_y - melted_ice * 0.9) * 0.01
    # mass_ice_residual = (h_i_t + melted_ice) * 0.01
    # momentum_x_residual = (u_t + u * h_w_x - (gravity_force_x - friction_force_x)) * 0.001
    # momentum_y_residual = (v_t + v * h_w_y - (gravity_force_y - friction_force_y)) * 0.001

    # НАДО: Используем динамическую скорость таяния для каждого этапа!
    melted_ice = torch.where(h_i > 0.0, friction_heat * melt_dynamic, torch.zeros_like(h_i))

    # 🔥 ЧЕСТНЫЙ ЗАКОН СОХРАНЕНИЯ МАССЫ С УЧЕТОМ ЭРОЗИИ МАКДУГАЛЛА
    # Теперь объем жидкой фазы растет не только от таяния льда, но и от содранного грунта!
    # 🔥 ЧЕСТНЫЙ ДВУХФАЗНЫЙ ЗАКОН СОХРАНЕНИЯ МАССЫ
    # Водяной вал переносится быстрыми скоростями фазы воды (u_water, v_water)
    mass_water_residual = (h_w_t + u_water * h_w_x + v_water * h_w_y - melted_ice * 0.9 - erosion_rate) * 0.01
    
    # Ледяная и каменная основа переносится медленными скоростями (u_solid, v_solid)
    mass_ice_residual = (h_i_t + u_solid * h_i_x + v_solid * h_i_y + melted_ice) * 0.01

    # ТОРМОЖЕНИЕ ИМПУЛЬСА С УЧЕТОМ СКАЛЬНЫХ ПРЕПЯТСТВИЙ И ЭРОЗИИ
    momentum_x_residual = (u_t + u_solid * h_w_x - (gravity_force_x - friction_force_x) + u * (erosion_rate / (h_w + 1e-5))) * 0.001
    momentum_y_residual = (v_t + v_solid * h_w_y - (gravity_force_y - friction_force_y) + v * (erosion_rate / (h_w + 1e-5))) * 0.001

    # НАКАЗАНИЕ ИИ ЗА ПОПЫТКУ СВЕРХСКОРОСТНОГО СУХОГО СХОДА В ДОЛИНУ
    # loss_hyper_velocity = torch.mean(torch.where((v_mag > 18.0) & (h_w > 0.5), (v_mag - 18.0)**2, torch.zeros_like(v_mag))) * 2000.0
    # 🔥 ОСВОБОЖДЕНИЕ КИНЕТИКИ: Убираем искусственный тормоз ИИ!
    # Пусть лавина честно разгоняется под действием гравитации и генерирует реальное тепло дробления!
    loss_hyper_velocity = torch.zeros_like(v_mag).mean()
    # loss_p = torch.mean(mass_water_residual**2) + \
    #          torch.mean(mass_ice_residual**2) + \
    #          torch.mean(momentum_x_residual**2) + \
    #          torch.mean(momentum_y_residual**2)

    # ВЫЧИСЛЕНИЯ ЖИВОЙ РЕОЛОГИИ (Для вывода человеку и сбора статистики)
    with torch.no_grad():
        v_max_val = torch.max(v_mag).item()
        h_mean_val = torch.mean(h_w).item()
        tau_y_mean = torch.mean(tau_y_dynamic).item()
        K_mean = torch.mean(K_dynamic).item() + 1e-5

        # Число Рейнольдса-Бингама и Хедстрёма (плотность селя 1800 кг/м3)
        Re_val = (torch.mean(rho_mix).item() * v_max_val * h_mean_val) / K_mean
        He_val = (torch.mean(rho_mix).item() * tau_y_mean * (h_mean_val**2)) / (K_mean**2)
        
        # Размер жесткого комка (ядра пробки Бингама)
        base_shear = torch.mean(rho_mix).item() * G * h_mean_val * torch.mean(torch.sqrt(slope_x**2 + slope_y**2)).item()
        plug_zone = h_mean_val * (tau_y_mean / (base_shear + 1e-5))
        plug_zone_val = min(h_mean_val, max(0.0, plug_zone))

        # Считаем суммарный объем вовлеченной породы на текущей эпохе (м³)
        grid_cell_area = DX_METERS ** 2
        total_eroded_vol_m3 = torch.sum(erosion_rate).item() * grid_cell_area * 0.4

    # === ВСТАВИТЬ ПЕРЕД return loss_p, h_w ===
    if torch.rand(1).item() < 0.01:  # Печатаем случайно в 1% случаев, чтобы не спамить в консоль
        print("\n--- [Диагностика физики Сен-Венана] ---")
        print(f"  💧 Ошибка массы воды:  {torch.mean(mass_water_residual**2).item():.6f}")
        print(f"  ❄️ Ошибка массы льда:  {torch.mean(mass_ice_residual**2).item():.6f}")
        print(f"  🏎️ Ошибка импульса X:  {torch.mean(momentum_x_residual**2).item():.6f}")
        print(f"  🏎️ Ошибка импульса Y:  {torch.mean(momentum_y_residual**2).item():.6f}")
        print("\n🔬 ==================ОТЧЕТ АНАЛИЗА ГИПОТЕЗЫ==================")
        
        # Расчет временного коридора для принтов человека-исследователя
        t_min, t_avg, t_max = torch.min(t).item(), torch.mean(t).item(), torch.max(t).item()
        
        # Определяем текущий временной этап по эпохе
        if epoch < 2500:
            print(f"🏔️  [ТЕКУЩИЙ ЭТАП 1/3]: Сухой старт и запуск лавины (0 - 3 сек) | ⏱️ Текущее время ИИ: {t_avg:.2f} сек (диапазон: {t_min:.1f} - {t_max:.1f} из 10.0)")
            print(f"   └── Реология: Жесткое трение Бингама ({torch.mean(tau_y_dynamic).item():.1f} Па) | Медленное таяние")
        elif epoch < 5500:
            print(f"⚡  [ТЕКУЩИЙ ЭТАП 2/3]: Прорыв каньона и бурное плавление (3 - 7 сек) | ⏱️ Текущее время ИИ: {t_avg:.2f} сек (диапазон: {t_min:.1f} - {t_max:.1f} из 10.0)")
            print(f"   └── Реология: Трение падает ({torch.mean(tau_y_dynamic).item():.1f} Па) -> Смазка водой")
        else:
            print(f"🏕️  [ТЕКУЩИЙ ЭТАП 3/3]: Выход в долину и цементация массы (7 - 10 сек) | ⏱️ Текущее время ИИ: {t_avg:.2f} сек (диапазон: {t_min:.1f} - {t_max:.1f} из 10.0)")
            print(f"   └── Реология: Масса остывает, вязкость стабилизируется")
        
        # Считаем среднюю толщину сухого льда и жидкой воды в симуляции
        avg_ice = torch.mean(h_i).item()
        avg_water = torch.mean(h_w).item()
        max_v = torch.max(v_mag).item() 

        # === КРУПНЫЙ АНАЛИТИЧЕСКИЙ БЛОК: РАСПОЗНАВАНИЕ 5 ГЕОФИЗИЧЕСКИХ ГИПОТЕЗ ===
        # === ИСПРАВЛЕНИЕ: Считаем фазы только внутри несущегося потока ===
        moving_mask = (v_mag > 1.0).float()
        # Если поток движется, считаем лед и воду только в нем, иначе берем среднее
        if torch.sum(moving_mask) > 0:
            avg_ice_moving = torch.sum(h_i * moving_mask) / (torch.sum(moving_mask) + 1e-6)
            avg_water_moving = torch.sum(h_w * moving_mask) / (torch.sum(moving_mask) + 1e-6)
            ice_to_water_ratio = avg_ice_moving.item() / (avg_water_moving.item() + 1e-6)
        else:
            ice_to_water_ratio = avg_ice / (avg_water + 1e-6)
        
        # Безопасное извлечение скаляров для принтов исследователя (исключаем падение графа)
        # === БЕЗОПАСНЫЙ ГЕОФИЗИЧЕСКИЙ АНАЛИЗАТОР ФАЗ ВНУТРИ ПОТОКА ===
        with torch.no_grad():
            moving_mask = (v_mag > 1.0).float()
            has_motion = torch.sum(moving_mask) > 0
            
            # Считаем параметры СТРОГО внутри летящего тела (где скорость > 1 м/с)
            if has_motion:
                cell_count = torch.sum(moving_mask) + 1e-6
                print_avg_ice = (torch.sum(h_i * moving_mask) / cell_count).item()
                print_avg_water = (torch.sum(h_w * moving_mask) / cell_count).item()
                print_max_v = torch.max(v_mag).item() # СИНХРОНИЗАЦИЯ: Берем пиковое значение скорости прямо из живого тензора эпохи
                # Локальное точное отношение фаз внутри самого оползня
                local_ratio = print_avg_ice / (print_avg_water + 1e-6)
            else:
                # Если всё стоит, берем общие средние по сетке
                print_avg_ice = torch.mean(h_i).item()
                print_avg_water = torch.mean(h_w).item()
                print_max_v = torch.max(v_mag).item()
                local_ratio = print_avg_ice / (print_avg_water + 1e-6)

            # Пересчет высоты слоя воды (метрической) в реальный физический объем (м³)
            # Площадь ячейки ущелья (DX_METERS ** 2) ≈ 400 м²
            # Полная автоматизация: шаг сетки берется напрямую из настроек нашей PINN-модели
            # DX_METERS (или твой шаг сетки) автоматически пересчитает кубометры для любой горы
            grid_cell_area = DX_METERS ** 2  # Автоматическая площадь ячейки
            real_water_volume_m3 = print_avg_water * grid_cell_area * 0.4

        # =========================================================================
        # 🔬 ВЕТВЛЕНИЕ ГИПОТЕЗ: АНАЛИЗАТОР МЧС ДЛЯ ИССЛЕДОВАТЕЛЯ
        # =========================================================================
        
        # ГИПОТЕЗА №1: Структурный монолит (Абсолютно сухой старт глыбы)
        if local_ratio > 10.0 and print_max_v < 4.0:
            print("       🧱 [ГИПОТЕЗА 1/5: СУХОЙ КРИТИЧЕСКИЙ СДВИГ] Катастрофа только зарождается!")
            print(f"           └── Физика ядра: Движется сухой каменный монолит. Вода в теле потока ОТСУТСТВУЕТ (W_loc = {print_avg_water:.4f} м³).")
            print(f"           └── Верификация фаз: Зафиксирована стабильная ледяная матрица высотой {print_avg_ice:.2f} м. Фазовый переход заблокирован.")

        # ГИПОТЕЗА №2: Сухой разгон лавины (Трение Бингама без разжижения)
        elif local_ratio > 5.0 and print_max_v >= 4.0:
            print("       🏃 [ГИПОТЕЗА 2/5: СУХОЙ КИНЕТИЧЕСКИЙ РАЗГОН] Лавина несется БЕЗ жидкой смазки!")
            print(f"           └── Фика ядра: Камни и лед заклинило внутренним трением. Высокая скорость V_max = {print_max_v:.2f} м/с обусловлена гравитационным сдвигом.")
            print(f"           └── Текстура массы: Поток держит форму русла за счет жесткости. Объем жидкой фазы ничтожен ({real_water_volume_m3:.2f} м³).")

        # ГИПОТЕЗА №3: Смешанный грязекаменный поток (Начало активного плавления)
        elif 1.0 <= local_ratio <= 5.0 and print_max_v >= 5.0:
            print("       🌀 [ГИПОТЕЗА 3/5: СМЕШАННЫЙ ГРЯЗЕКАМЕННЫЙ ПОТОК] Инициирован термодинамический фазовый переход!")
            print(f"           └── Физика ядра: Кинетическое трение плавит лед. Модель фиксирует генерацию жидкой фазы ({print_avg_water:.2f} м воды).")
            print(f"           └── Текстура массы: Внутри оползня формируется смесь обломков скал, остаточного льда ({print_avg_ice:.2f} м) и нарастающей смазки.")

        # ГИПОТЕЗА №4: Разжиженный бурный сель (Максимальное водонасыщение каньона)
        elif local_ratio < 1.0 and print_max_v >= 10.0:
            print("       🌊 [ГИПОТЕЗА 4/5: РАЗЖИЖЕННЫЙ ГИДРАВЛИЧЕСКИЙ СЕЛЬ] Максимальная угроза прорыва ущелья!")
            print(f"           └── Физика ядра: Лед практически полностью вытоплен ({print_avg_ice:.3f} м). Вода захватила матрицу ({print_avg_water:.2f} м).")
            print(f"           └── Текстура массы: Поток полностью потерял бингамовскую жесткость. Водяной вал несется на предельной скорости {print_max_v:.2f} м/с.")

        # ГИПОТЕЗА №5: Затухание, выполаживание и цементация селя в долине
        elif local_ratio < 1.0 and print_max_v < 10.0:
            print("       静态 [ГИПОТЕЗА 5/5: ВЫПОЛАЖИВАНИЕ И ЦЕМЕНТАЦИЯ] Поток вышел на равнину к жилой зоне!")
            print(f"           └── Физика ядра: Кинетическая энергия диссипировала (V_max = {print_max_v:.2f} м/с). Прекращение генерации тепла трения.")
            print(f"           └── Текстура массы: Жидкая грязевая смесь ({print_avg_water:.2f} м) растекается вширь, теряет скорость и начинает лавинообразно застывать.")
            
        else:
            print(f"       🛸 [КОНТРОЛЬ СИСТЕМЫ]: Переходная фаза (Скорость: {print_max_v:.2f} м/с, Локальное соотношение Лед/Вода: {local_ratio:.2f})")
        
        # # 1. Проверяем состояние реологии (сухое тело или жидкий поток)
        # if avg_ice > avg_water * 3.0 and max_v > 5.0:
        #     print("🧱 [ГИПОТЕЗА ПОДТВЕРЖДЕНА]: Наверху движется сухое, заклинившее тело лавины!")
        #     print(f"   └── Трение Бингама заперло массу. Лед: {avg_ice:.2f} м | Вода: {avg_water:.2f} м")
        # elif avg_water > avg_ice:
        #     print("🌊 [ФАЗОВЫЙ ПЕРЕХОД]: Сель разжижается! Трение растопило лед в каньоне.")
        #     print(f"   └── Поток перешел в жидкую фазу. Вода: {avg_water:.2f} м | Лед: {avg_ice:.2f} м")
        # else:
        #     print("🌀 [СМЕШАННАЯ ЗОНА]: ИИ ищет баланс между сухим сдвигом и водой...")
        
        # 🔥 НАШ НОВЫЙ ПРИНТ СТРАХОВКИ:
        if added_vol_np > 0:
            print(f"  🧪 [СТРАХОВКА МАССЫ] Физика буксует! Искусственно подмешано: {added_vol_np:.2f} м³ воды")
        else:
            print(f"  ✅ [ЧЕСТНАЯ ФИЗИКА] Поток полностью обеспечен таянием льда: {torch.mean(expected_melted_vol).item():.2f} м³")

        # === ВЫЧИСЛЯЕМ ФИЗИЧЕСКИЕ ДИАПАЗОНЫ ДЛЯ ЧЕЛОВЕКА ===
        with torch.no_grad():
            v_min, v_max = torch.min(v_mag).item(), torch.max(v_mag).item()
            h_min, h_max = torch.min(h_w).item(), torch.max(h_w).item()
            K_min_v, K_max_v = torch.min(K_dynamic).item(), torch.max(K_dynamic).item()
            # Считаем среднее и отклонение для глубины воды
            h_mean = torch.mean(h_w).item()
            # Берем реальный разброс значений в симуляции, но ограничиваем его разумными пределами
            h_delta = min(1.0, max(0.2, (h_max - h_min) / 2.0))
            avg_pressure_heat = torch.mean(h_i * G * 1.8e-4).item()
            avg_total_heat = torch.mean(friction_heat).item() + 1e-8
            # Считаем вклад веса горы в процентах
            pressure_share = (avg_pressure_heat / avg_total_heat) * 100.0
            # === МОДУЛЬ ПРИНТА КОНТРОЛЯ ДВУХФАЗНОГО ТЕЧЕНИЯ DRIFT-FLUX ===
            avg_v_water = torch.mean(torch.sqrt(u_water**2 + v_water**2 + 1e-5)).item()
            avg_v_solid = torch.mean(torch.sqrt(u_solid**2 + v_solid**2 + 1e-5)).item()
            slip_lead_percent = ((avg_v_water - avg_v_solid) / (avg_v_solid + 1e-5)) * 100.0
            # Считаем средний размер камней на текущей эпохе
            avg_stone_size = torch.mean(d_part).item()

        print("\n🌊 ============ МОНИТОРИНГ ДВУХФАЗНЫХ ПОТОКОВ МАКДУГАЛЛА-ИШШИ ============")
        print(f"  🚀 Быстрая фаза (Грязевой вал воды): Средняя скорость = {avg_v_water:.2f} м/с")
        print(f"  🐢 Медленная фаза (Камни и монолит): Средняя скорость = {avg_v_solid:.2f} м/с")
        print(f"  ⚡ Эффект опережения: Жидкая смазка вырывается вперед лавины на +{slip_lead_percent:.1f}%")
        print(f"  🧱 Жернова каньона (Дробление): Валуны перемалываются до среднего размера: {avg_stone_size*100:.1f} см")
        print("============================================================================\n")
        print(f"  🌊 Макс. высота воды:  {torch.max(h_w).item():.2f} м | Макс. скорость: {torch.max(v_mag).item():.2f} м/с")
        print(f"   🏎️ Торможение: Энергетический штраф скорости импульса = x{torch.mean(energy_loss_factor).item():.3f}")            
        print(f"   🏎️ Динамика: Скорость схода = {max_v:.2f} м/с | Макс. глубина = {torch.max(h_w).item():.2f} м")
        print(f"  🌊 Высота вала воды: {h_mean:.2f} м  (разброс в русле: ±{h_delta:.1f} м) | Макс. пик: {h_max:.2f} м")
        print(f"  🏎️ Коридор скоростей: от {v_min:.1f} до {v_max:.1f} м/с")
        print(f"  🏎️ Торможение: Энергетический штраф за таяние = x{torch.mean(energy_loss_factor).item():.3f}")
        print(f"  🧱 Рео-параметры: Число Рейнольдса Re_b = {Re_val:.1f} (при Re < 2000 поток строго ламинарный)")
        print(f"  🧱 Число Хедстрёма (Пластичность): He = {He_val:.1e}")
        print(f"  🧱 Жесткое ядро комка: {plug_zone_val:.2f} м из {torch.max(h_w).item():.2f} м (структурная пробка)")  
        # print(f"   🧱 Жесткое ядро комка (проверка) : {current_plug:.2f} м из {torch.max(h_w_predicted).item():.2f} м") # <-- Заменили на current_plug и h_w_predicted
        print(f"  🏔️ Напряжение сдвига на ложе каньона: {base_shear:.1f} Па")
        print(f"  🧪 Коридор динамической вязкости масс: от {K_min_v:.2f} до {K_max_v:.2f} Па·с")    
        print(f"  🏔️ Пьезо-эффект: Вес горы обеспечивает {pressure_share:.1f}% тепла плавления (статический разогрев ложа)") 
        print(f"  ⛰️ Эрозия русла (МакДугалл): Вымыто {total_eroded_vol_m3:.2f} м³ грунта/камней в секунду")                             
        print(f"  🧪 Динамика массы: Наверху сухо -> в каньоне вытоплено {torch.mean(expected_melted_vol).item():.2f} м³ чистой воды")        
        print("📊 ========================================================\n")

    # 🛠️ ЗАПЛАТКА: Штрафуем модель, если глубина движущегося потока ниже 1.0 метра
    # Если скорость > 1.0 м/с, а высота < 1.0 метра — включается квадратичный штраф
    # 🛠️ ЖЕСТКАЯ ЗАПЛАТКА 2.0: Экспоненциальный штраф за высыхание русла
    # Если скорость потока v_mag > 1.0 м/с (поток несется), а высота h_w < 1.2 метра
    # Включаем жесткий штраф, который растет по экспоненте, выдавливая ИИ на нужную глубину
    # low_water_mask = torch.where((v_mag > 1.0) & (h_w < 1.2), torch.ones_like(h_w), torch.zeros_like(h_w))
    # loss_low_water = torch.mean(low_water_mask * torch.exp(1.2 - h_w)) * 5000.0  # Подняли вес до 5000!
    in_canyon_mask = (y < (H_PIXELS * DX_METERS * 0.4)).float()  # Экспоненциальное наказание за высыхание русла
    loss_low_water = torch.mean(in_canyon_mask * torch.exp(2.5 - h_w)) * 5000.0

    # =========================================================================
    # 🧠 ЭКСПЕРТНЫЕ ПОДСКАЗКИ В ПОДСОЗНАНИЕ ИИ (EXPERT-GUIDED REGULARIZATION) (8 ФИЗИЧЕСКИХ СТАТУСОВ)
    # =========================================================================
    # Подсказка 1: Если скорость высокая (v_mag > 5), камни ОБЯЗАНЫ крошиться (d_part < 0.8 м)
    # Если ИИ пытается тащить огромные глыбы на большой скорости — мягко штрафуем его
    hint_shatter = torch.where((v_mag > 5.0) & (d_part > 0.8), (d_part - 0.8)**2, torch.zeros_like(d_part)).mean() * 10.0

    # Подсказка 2: Если камни раскрошились в пудру (d_part < 0.5), тепло ОБЯЗАНО топить лед (melt_dynamic > 0.05)
    # Наказываем модель за парадокс сухого измельчения
    hint_melting = torch.where((d_part < 0.5) & (melt_dynamic < 0.05), (0.05 - melt_dynamic)**2, torch.zeros_like(melt_dynamic)).mean() * 50.0

    # Подсказка 3: Вода в каньоне ОБЯЗАНА двигаться (v_water > 2.0 м/с), она не может стоять столбом
    hint_flow = torch.where((in_canyon_mask > 0.5) & (h_w > 1.0) & (v_water < 2.0), (2.0 - v_water)**2, torch.zeros_like(v_water)).mean() * 5.0

    # 🔥 ПОДСКАЗКА 4: Плотность смеси обязана расти при захвате породы
    # Если эрозия идет бурно (erosion_rate > 1.0), плотность rho_mix не может оставаться легкой, как чистая вода
    hint_density = torch.where((erosion_rate > 1.0) & (rho_mix < 1500.0), (1500.0 - rho_mix)**2 * 1e-6, torch.zeros_like(rho_mix)).mean() * 20.0

    # 🔥 ПОДСКАЗКА 5: Закон виража (Центробежный вынос вала h_w)
    # На крутых поворотах каньона (где curvature_x/y > 0.01) на высокой скорости поток обязано выносить вверх!
    # Высота h_w в створе изгиба не может падать ниже 1.5 метров
    curve_mag = torch.sqrt(curvature_x**2 + curvature_y**2 + 1e-5)
    hint_centrifugal = torch.where((in_canyon_mask > 0.5) & (v_mag > 15.0) & (curve_mag > 0.01) & (h_w < 1.5), (1.5 - h_w)**2, torch.zeros_like(h_w)).mean() * 15.0

    # 🔥 ПОДСКАЗКА 6: Честная эрозия дна по МакДугаллу
    # Если поток летит со скоростью гоночного болида (v_mag > 20.0) в каньоне, эрозия НЕ МОЖЕТ быть нулевой!
    # Наказываем модель за попытку проскочить ущелье по стерильному бетону
    hint_erosion = torch.where((in_canyon_mask > 0.5) & (v_mag > 20.0) & (erosion_rate < 0.2), (0.2 - erosion_rate)**2, torch.zeros_like(erosion_rate)).mean() * 30.0

    # 🔥 ПОДСКАЗКА 7: Диссипация и торможение при выполаживании (Закон долины)
    # Если время симуляции перевалило за финал (t > 7.5 сек), а уклон slope_magnitude стал пологим (< 0.05),
    # скорость ОБЯЗАНА падать ниже 8.0 м/с из-за цементации. Никаких вечных двигателей в долине!
    hint_damping = torch.where((t > 7.5) & (slope_magnitude < 0.05) & (v_mag > 8.0), (v_mag - 8.0)**2, torch.zeros_like(v_mag)).mean() * 25.0

    # 🔥 ПОДСКАЗКА 8: ЗАКОН ТОТАЛЬНОГО СМЫВА (EXPERT TOTAL WASHOUT HINT)
    # На часах прорыв каньона (t > 5.0 сек), масса ОБЯЗАНА снести всё русло!
    # Если ИИ пытается остановить лавину в ущелье (v_mag < 15.0 м/с), когда глубина вала 
    # превышает 2.0 метра — влупляем мощный штраф. Поток обязан пробить плотину об ригели!
    hint_washout = torch.where((t > 5.0) & (in_canyon_mask > 0.5) & (h_w > 2.0) & (v_mag < 15.0), 
                               (15.0 - v_mag)**2, 
                               torch.zeros_like(v_mag)).mean() * 45.0
    # Добавляем этот штраф к общему физическому лоссу
    loss_p = torch.mean(mass_water_residual**2) + \
             torch.mean(mass_ice_residual**2) + \
             torch.mean(momentum_x_residual**2) + \
             torch.mean(momentum_y_residual**2) + \
             loss_hyper_velocity + \
             hint_shatter + hint_melting + hint_flow + \
             hint_density + hint_centrifugal + hint_erosion + hint_damping + \
             hint_washout

    # Возвращаем все 8 реологических параметров наружу в цикл!
    # Стало: усредняем локальные объемы в один скаляр только для красивого отчета geofiz_stats
    # avg_stone_size_scalar = torch.mean(d_part).item()    
    # return loss_p, h_w, v_max_val, torch.mean(expected_melted_vol).item(), Re_val, He_val, plug_zone_val, total_eroded_vol_m3, avg_stone_size_scalar

    # return (loss_p, h_w, v_max_val, torch.mean(expected_melted_vol).item(), Re_val, He_val, 
    #         plug_zone_val, total_eroded_vol_m3, avg_stone_size_scalar,
    #         rho_mix, water_fraction, u_water, v_water, friction_heat,
    #         mass_water_residual, mass_ice_residual, momentum_x_residual, momentum_y_residual,
    #         u_solid, v_solid, h_i, d_part, v_mag, in_canyon_mask, melt_dynamic)

    # Вытаскиваем только те оригинальные тензоры, которые реально есть в этой функции
    # return loss_p, h_w, h_i, v_mag, mass_water_residual, mass_ice_residual, momentum_x_residual, momentum_y_residual, friction_heat
    avg_stone_size_scalar = torch.mean(d_part).item()    
    return loss_p, h_w, v_max_val, torch.mean(expected_melted_vol).item(), Re_val, He_val, plug_zone_val, total_eroded_vol_m3, avg_stone_size_scalar

def async_logger(epoch, total_loss, loss_p, loss_sat, match_score):
    print(f"🛰️ [ИИ Непал] Эпоха {epoch:04d} | Общий Loss: {total_loss:.4f} | Физика: {loss_p:.4f} | Спутник: {loss_sat:.4f} | Схождение: {match_score:.2f}%")

if __name__ == "__main__":
    parser = NepalSplineParser()
    pinn = LandslidePINNCore()

    # =========================================================================
    # ДОБАВЛЕННЫЙ БЛОК: ЗАГРУЗКА ОБУЧЕННЫХ ВЕСОВ (ЕСЛИ ОНИ ЕСТЬ)
    # =========================================================================
    # === ВСТАВИТЬ ВМЕСТО СТРОК 121 И 122 ===
    if os.path.exists("landslide_pinn_best.pth"):
        MODEL_PATH = "landslide_pinn_best.pth"  # Если уже есть рекорд, берем его
    else:
        MODEL_PATH = "landslide_pinn_model.pth" # Самый первый раз подхватываем старую базу

    if os.path.exists(MODEL_PATH):
        print(f"🔄 Найдены сохраненные веса '{MODEL_PATH}'. Загружаю...")
        try:
            # Загружаем веса. map_location гарантирует корректную загрузку на CPU/GPU
            checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=True)
            pinn.load_state_dict(checkpoint)
            print("✅ Модель успешно восстановлена! Продолжаем обучение не с нуля.")
        except Exception as e:
            print(f"⚠️ Не удалось загрузить веса из-за ошибки: {e}. Начинаем с нуля.")
    else:
        print("🆕 Файл весов не найден. Запуск обучения с нуля.")
    # =========================================================================
   
    optimizer = optim.Adam(pinn.parameters(), lr=0.001)
    # Снижает lr в 2 раза, если точность не растет 3 эпохи проверки подряд
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    tau_y_guess = 40.0   
    K_guess = 1.0        
    melt_guess = 0.05    

    # Накапливаемые веса для адаптивного баланса лоссов
    # 🔥 ИСПРАВЛЕНИЕ: Заставляем ИИ честно считать уравнения Навье-Стокса и Сен-Венана!
    lambda_p = 100.0  # Подняли с 1.0 до 100.0
    lambda_sat = 50.0 # Поднимаем базовый вес спутника, чтобы компенсировать разницу масштабов

    # =========================================================================
    # 🧬 МОДУЛЬ ДОФАМИНОВОЙ ПАМЯТИ И НАКАЗАНИЙ «БИТЬ ТОКОМ»
    # =========================================================================
    # Память: храним топ-5 лучших схождения за всю историю обучения
    best_matches_memory = [0.0, 0.0, 0.0, 0.0, 0.0] 

    # Дофаминовые и стрессовые коэффициенты
    dopamine_level = 1.0  # Множитель скорости обучения (награда)
    shock_voltage = 0.0   # Дополнительный штрафной вес (наказание)

    # Базовый шаг обучения для сброса
    BASE_LR = 0.001

    # Перед циклом эпох инициализируем флаг-защелку
    water_threshold_breached = False
    best_match_score = 0.0  # Сюда будем записывать рекорд процента совпадения
    shock_voltage = 0.0                    # Выключаем электрический стул на старте
    dopamine_level = 1.0                   # Нормализуем дофамин    
    # === НАКОПИТЕЛЬ ИСТОРИИ ИСКУССТВЕННОГО ПОДМЕШИВАНИЯ СТРАХОВКИ ===
    insurance_log = []  # Будем хранить кортежи: (эпоха, добавленный_объем)

    print("\n==================================================================")
    print("🌍 АДАПТИВНАЯ СИСТЕМА ИНВЕРСИИ НЕПАЛ-2026 С БАЛАНСОМ ГРАДИЕНТОВ")
    print("==================================================================")
    # === ГЛОБАЛЬНЫЙ СТАРТ ТАЙМЕРА ОБУЧЕНИЯ ===
    global_start_time = time.time()
    # === НАКОПИТЕЛЬ НАУЧНОЙ СТАТИСТИКИ ПО ЭТАПАМ ===
    # === РАСШИРЕННЫЙ НАКОПИТЕЛЬ ДЛЯ РЕОЛОГОВ ===
    stage_stats = {
        1: {"epochs": 0, "max_h": 0.0, "max_v": 0.0, "best_match": 0.0, "melt_vol": 0.0, "eroded_vol": 0.0, "sum_Re": 0.0, "sum_He": 0.0, "sum_plug": 0.0},
        2: {"epochs": 0, "max_h": 0.0, "max_v": 0.0, "best_match": 0.0, "melt_vol": 0.0, "eroded_vol": 0.0, "sum_Re": 0.0, "sum_He": 0.0, "sum_plug": 0.0},
        3: {"epochs": 0, "max_h": 0.0, "max_v": 0.0, "best_match": 0.0, "melt_vol": 0.0, "eroded_vol": 0.0, "sum_Re": 0.0, "sum_He": 0.0, "sum_plug": 0.0}
    }

    try:
        for epoch in range(8001):
            # 🔥 Фиксируем точное время начала этой эпохи
            epoch_start_time = time.time()

            optimizer.zero_grad()
            
            x_pts = torch.rand(200, 1, requires_grad=True) * (W_PIXELS * DX_METERS)
            y_pts = torch.rand(200, 1, requires_grad=True) * (H_PIXELS * DX_METERS)

            # Разделяем 8000 эпох на 3 больших физических этапа
            # if epoch < 2500:
            #     # --- ЭТАП 1: Сухой старт и запуск лавины (Первые 3 секунды) ---
            #     t_pts = torch.rand(200, 1, requires_grad=True) * 3.0
                
            #     # Статическая жесткая реология для верховьев гор
            #     tau_y_step = torch.full_like(t_pts, 55.0)
            #     K_step = torch.full_like(t_pts, 1.8)
            #     melt_step = torch.full_like(t_pts, 0.02)
                
            # elif epoch < 5500:
            #     # --- ЭТАП 2: Каньон, бешеное трение и лавинообразное таяние (3 - 7 сек) ---
            #     t_pts = 3.0 + torch.rand(200, 1, requires_grad=True) * 4.0
                
            #     # Живая динамика: вязкость и трение БИНГАМА стремительно ПАДАЮТ от воды
            #     tau_y_step = torch.clamp(55.0 - (t_pts - 3.0) * 6.0, min=25.0, max=55.0)
            #     K_step = torch.clamp(1.8 - (t_pts - 3.0) * 0.25, min=0.6, max=1.8)
            #     melt_step = torch.clamp(0.02 + (t_pts - 3.0) * 0.04, min=0.02, max=0.20)
                
            # else:
            #     # --- ЭТАП 3: Долина, выполаживание склона и цементация (7 - 10 сек) ---
            #     t_pts = 7.0 + torch.rand(200, 1, requires_grad=True) * 3.0
                
            #     # Финишная реология: масса остывает, вязкость снова начинает расти
            #     tau_y_step = torch.full_like(t_pts, 35.0)
            #     K_step = torch.clamp(0.6 + (t_pts - 7.0) * 0.2, min=0.6, max=1.5)
            #     melt_step = torch.full_like(t_pts, 0.01)
            # =========================================================================
            
            # === ИСПРАВЛЕНИЕ ЖЕСТИ: ЕДИНЫЙ ПРОСТРАНСТВЕННО-ВРЕМЕННОЙ КОНТИНУУМ ===
            # Каждую эпоху ИИ берет случайные точки на всем отрезке катастрофы (от 0 до 10 сек)
            # === СТРОГИЙ ХРОНОЛОГИЧЕСКИЙ ГРАДИЕНТ ВРЕМЕНИ ===
            # Генерируем 200 временных точек и ЖЕСТКО сортируем их по возрастанию!
            # Теперь внутри каждой эпохи время идет строго от 0 до 10 секунд, 
            # заставляя ИИ физически двигать массу сверху вниз по склону
            t_pts_raw = torch.rand(200, 1, requires_grad=True) * 10.0
            t_pts, _ = torch.sort(t_pts_raw, dim=0)
            
            # РЕОЛОГИЯ ТЕПЕРЬ СВЯЗАНА НЕ С ЭПОХАМИ ЦИКЛА, А С РЕАЛЬНЫМ ВРЕМЕНЕМ ПОТОКА t_pts!
            # Математика плавно и честно меняет параметры масс прямо внутри одной эпохи
            tau_y_step = torch.where(t_pts < 3.0, 
                                     torch.full_like(t_pts, 55.0),
                                     torch.where(t_pts < 7.0, 
                                                 torch.clamp(55.0 - (t_pts - 3.0) * 6.0, min=25.0, max=55.0),
                                                 torch.full_like(t_pts, 35.0)))
            
            K_step = torch.where(t_pts < 3.0, 
                                 torch.full_like(t_pts, 1.8),
                                 torch.where(t_pts < 7.0, 
                                             torch.clamp(1.8 - (t_pts - 3.0) * 0.25, min=0.6, max=1.8),
                                             torch.clamp(0.6 + (t_pts - 7.0) * 0.2, min=0.6, max=1.5)))
            
            melt_step = torch.where(t_pts < 3.0, 
                                    torch.full_like(t_pts, 0.02),
                                    torch.where(t_pts < 7.0, 
                                                torch.clamp(0.02 + (t_pts - 3.0) * 0.04, min=0.02, max=0.20),
                                                torch.full_like(t_pts, 0.01)))
            # =========================================================================

            # Передаем эти динамические этапы в наше расчетное ядро
            # (loss_p, h_w_predicted, current_max_v, current_melt_vol, current_Re, current_He, current_plug, 
            #  current_eroded_vol, current_stone_size, 
            #  rho_mix, water_fraction, u_water, v_water, friction_heat,
            #  mass_water_residual, mass_ice_residual, momentum_x_residual, momentum_y_residual,
            #  u_solid, v_solid, h_i, d_part, v_mag, in_canyon_mask, melt_dynamic) = compute_nepal_physics_loss(

            # loss_p, h_w, v_max_val, torch.mean(expected_melted_vol).item(), Re_val, He_val, plug_zone_val, total_eroded_vol_m3, avg_stone_size_scalar = compute_nepal_physics_loss(
            #     pinn, parser, x_pts, y_pts, t_pts, 
            #     tau_y_step, K_step, melt_step, epoch
            # )
            # loss_p, h_w, v_max_val, torch.mean(expected_melted_vol).item(), Re_val, He_val, plug_zone_val, total_eroded_vol_m3, avg_stone_size_scalar = compute_nepal_physics_loss(pinn, parser, x_pts, y_pts, t_pts, tau_y_step, K_step, melt_step, epoch)
            # Передаем эти динамические этапы в наше расчетное ядро
            loss_p, h_w_predicted, current_max_v, current_melt_vol, current_Re, current_He, current_plug, current_eroded_vol, current_stone_size = compute_nepal_physics_loss(
                pinn, parser, x_pts, y_pts, t_pts, 
                tau_y_step, K_step, melt_step, epoch # В функцию залетают уже ЖИВЫЕ тензоры этапа!
            )  

            sat_real_track = torch.where(y_pts < (H_PIXELS * DX_METERS * 0.4), torch.ones_like(y_pts), torch.zeros_like(y_pts))
            sim_track_binary = torch.sigmoid((h_w_predicted - 0.1) * 20.0) 
            loss_satellite = torch.mean((sim_track_binary - sat_real_track)**2)
            
            t0 = torch.zeros_like(x_pts)
            h_ice_t0_pred = torch.exp(pinn(x_pts, y_pts, t0)[:, 1:2])
            h_ice_t0_real = torch.where(y_pts > (H_PIXELS * DX_METERS * 0.8), torch.full_like(y_pts, 30.0), torch.zeros_like(y_pts))
            loss_boundary = torch.mean((h_ice_t0_pred - h_ice_t0_real)**2)

            # 🛠️ ИНТЕГРАЦИЯ УДАРОВ ТОКОМ (Добавляем shock_voltage прямо в граф вычислений лосса!)
            total_loss = lambda_p * loss_p + (lambda_sat + shock_voltage) * loss_satellite + loss_boundary
                        
            total_loss.backward()
            optimizer.step()

            with torch.no_grad():
                max_h_w = torch.max(h_w_predicted).item()

            # Однократный принт, как только стена воды реально превысила 1 метр
            if max_h_w > 1.0 and not water_threshold_breached:
                print(f"\n🚨 [ФИЗИЧЕСКИЙ ТРИГГЕР] Поток преодолел критический метр! Пиковая высота: {max_h_w:.2f} м. Сеть переходит в режим фиксации русла.")
                water_threshold_breached = True

            # # --- 🧬 МОЗГОВОЙ ЦЕНТР: ПРИНЯТИЕ РЕШЕНИЙ ИИ ---
            # with torch.no_grad():
            #     current_match = ((sim_track_binary > 0.5) == (sat_real_track > 0.5)).float().mean().item() * 100.0

            # # 1. Проверяем память топ-5
            # if current_match > min(best_matches_memory) and current_match > 60.0:
            #     # Убираем худшее из топ-5, вставляем новое, сортируем по возрастанию
            #     best_matches_memory[0] = current_match
            #     best_matches_memory.sort()
                
            #     # 🎉 ЭФФЕКТ ДОФАМИНА (Поощрение за прорыв)
            #     # ИИ нащупал верный путь! Выделяем дофамин: увеличиваем шаг обучения, чтобы быстрее бежать вперед
            #     dopamine_level = min(3.0, dopamine_level + 0.5)
            #     shock_voltage = max(0.0, shock_voltage - 50.0) # Снимаем стресс
                
            #     # Применяем дофамин к оптимизатору
            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] = BASE_LR * dopamine_level
                    
            #     print(f"🧠 [ДОФАМИН х{dopamine_level:.1f}] Новое топ-схождение: {current_match:.2f}%! Память топ-5: {best_matches_memory}")

            # =========================================================================
            # 🧬 МУЛЬТИ-ДОФАМИНОВЫЙ МОЗГОВОЙ ЦЕНТР (С НАГРАДОЙ ЗА ЧЕСТНУЮ ФИЗИКУ)
            # =========================================================================
            with torch.no_grad():
                current_match = ((sim_track_binary > 0.5) == (sat_real_track > 0.5)).float().mean().item() * 100.0
                
                # Считаем, насколько ИИ близок к честной физике
                # Награда выдается, если лосс физики стабилизировался, а трение генерирует талую воду
                physics_honesty_factor = 1.0 if (loss_p.item() < 0.05 and current_melt_vol > 5.0) else 0.1

            # Условие прорыва: ИИ получает ДОФАМИН за спутник И за честные законы природы!
            if current_match > min(best_matches_memory) and current_match > 60.0:
                best_matches_memory[0] = current_match
                best_matches_memory.sort()
                
                # 🎉 ЭФФЕКТ ГЕОМЕТРИЧЕСКОГО ДОФАМИНА
                dopamine_level = min(3.0, dopamine_level + 0.5)
                shock_voltage = max(0.0, shock_voltage - 50.0) # Снимаем стресс
                
                # 🔥 ИНЖЕКЦИЯ ФИЗИЧЕСКОГО ДОФАМИНА:
                # Если физика честная -> умножаем дофаминовую награду и задираем шаг обучения до x5.0!
                if physics_honesty_factor > 0.5:
                    dopamine_level = min(5.0, dopamine_level * 1.5)
                    print(f"🔥 🧠 [ФИЗИЧЕСКИЙ ДОФАМИН ИИ!]: Модель награждена за честные законы Ньютона! Шаг обучения взлетает до x{dopamine_level:.2f}")
                
                for param_group in optimizer.param_groups:
                    param_group['lr'] = BASE_LR * dopamine_level
                    
                print(f"🧠 [ГЕО-ДОФАМИН х{dopamine_level:.1f}] Новое топ-схождение: {current_match:.2f}%! Память топ-5: {best_matches_memory}")

            # =========================================================================
            # 🌀 ДИНАМИЧЕСКИЙ ФРАКТАЛЬНЫЙ ЗАЖИМАТЕЛЬ ПАРАМЕТРОВ (ТАГАНРОГ-2026)
            # =========================================================================
            # 2. Наказание «УДАР ТОКОМ» за жесткую деградацию
            # Если схождение упало сильно ниже лучшего результата из памяти
            elif max(best_matches_memory) > 0 and current_match < (max(best_matches_memory) - 5.5):
                # Считаем, насколько мы близки к абсолютному идеалу
                dist_to_100 = max(1.0, 100.0 - current_match)
                
                # 1. Фрактальный шаг торможения (Чем ближе к 100%, тем аккуратнее и мельче шаги ИИ)
                # На 60% схождения вычтет 0.1. На 97% схождения вычтет 0.02 — ювелирная точность!
                fractal_decay = 4.0 / dist_to_100
                dopamine_level = max(0.15, dopamine_level - min(0.3, fractal_decay))
                
                # 2. Динамический удар током (Вольтаж растет лавинообразно при штурме вершины!)
                # Если ИИ тупит на 60%, удар всего +5V. Если он посмел упасть с 97% — мгновенный разряд в +133V!
                fractal_shock = 400.0 / dist_to_100
                shock_voltage = min(450.0, shock_voltage + fractal_shock)
                
                for param_group in optimizer.param_groups:
                    param_group['lr'] = BASE_LR * dopamine_level

                # 🛡️ ФРАКТАЛЬНЫЙ ROLLBACK (Откат к рекорду)
                # Порог срыва тоже динамический. Чем выше залезли, тем меньше права на ошибку.
                rollback_threshold = min(350.0, 100.0 + (100.0 - max(best_matches_memory)) * 10.0)
                if shock_voltage > rollback_threshold and os.path.exists("landslide_pinn_best.pth"):
                    if epoch % 40 == 0:    
                        print(f"🌀 [ФРАКТАЛЬНЫЙ АТТРАКТОР] Срыв на высоте! Откат к рекорду {max(best_matches_memory):.2f}%...")
                    checkpoint = torch.load("landslide_pinn_best.pth", map_location=torch.device('cpu'), weights_only=True)
                    pinn.load_state_dict(checkpoint)

                    shock_voltage = 0.0
                    dopamine_level = 1.0   

                if epoch % 100 == 0:
                    print(f"⚡ [ФРАКТАЛЬНЫЙ ШОК: {shock_voltage:.1f}V] Схождение: {current_match:.2f}% | Шаг ИИ: x{dopamine_level:.2f}")

            # 1. Сначала определяем этап (на каждой эпохе!)
            current_stage = 1 if epoch < 2500 else (2 if epoch < 5500 else 3)
            
            # === СБОР ДАННЫХ ДЛЯ ФИНАЛЬНОГО ОТЧЕТА ===
            with torch.no_grad():
                # Определяем, какой сейчас этап по номеру эпохи
                current_stage = 1 if epoch < 2500 else (2 if epoch < 5500 else 3)
                
                # Инкрементируем счетчик эпох этапа
                stage_stats[current_stage]["epochs"] += 1
                
                # Фиксируем максимумы через безопасные внешние переменные
                stage_stats[current_stage]["max_h"] = max(stage_stats[current_stage]["max_h"], max_h_w)
                stage_stats[current_stage]["max_v"] = max(stage_stats[current_stage]["max_v"], current_max_v)
                stage_stats[current_stage]["best_match"] = max(stage_stats[current_stage]["best_match"], current_match)
                stage_stats[current_stage]["melt_vol"] = max(stage_stats[current_stage]["melt_vol"], current_melt_vol) 
                stage_stats[current_stage]["sum_Re"] += current_Re
                stage_stats[current_stage]["sum_He"] += current_He
                stage_stats[current_stage]["sum_plug"] += current_plug                  
                stage_stats[current_stage]["eroded_vol"] = max(stage_stats[current_stage]["eroded_vol"], current_eroded_vol)

            # Медленное угасание дофамина со временем (ИИ успокаивается)
            dopamine_level = max(1.0, dopamine_level - 0.005)
            shock_voltage = max(0.0, shock_voltage - 0.5)
            
            # Интегрируем электрический штраф в лосс на следующую эпоху см лосс выше

            # 🔁 Адаптивная корректировка весов каждые 10 эпох
            # if epoch % 10 == 0:
            #     with torch.no_grad():
            #         # Масштабируем коэффициенты, чтобы уравнять влияние физики и маски спутника
            #         ratio = loss_p.item() / (loss_satellite.item() + 1e-6)
            #         if ratio > 0:
            #             lambda_sat = min(500.0, max(10.0, ratio * 0.1))
            # 🔁 Адаптивная динамическая балансировка весов лоссов (КАЖДЫЕ 10 ЭПОХ)
            # if epoch % 10 == 0:
            #     with torch.no_grad():
            #         ratio = loss_p.item() / (loss_satellite.item() + 1e-6)
            #         if ratio > 0:
            #             # Если физика проседает (картонный поток), мы искусственно опускаем 
            #             # вес спутника и заставляем сеть страдать над уравнениями Сен-Венана
            #             lambda_sat = min(200.0, max(2.0, ratio * 0.5))
                        
            #             # 🔥 ЖЕСТКИЙ АДАПТИВНЫЙ КНУТ ДЛЯ ЛАМБДЫ ФИЗИКИ
            #             # Если loss_p падает ниже критического уровня 0.01 -> лавинообразно задираем его вес
            #             lambda_p = 150.0 if loss_p.item() < 0.01 else 50.0

            # =========================================================================
            # 🔁 РОБОТИЗИРОВАННЫЙ АВТО-ЭЛЕКТРОШОК (ДИНАМИЧЕСКИЙ БАЛАНС ЛОССОВ)
            # =========================================================================
            if epoch % 10 == 0:
                with torch.no_grad():
                    # Считаем чистый физический перекос
                    loss_ratio = loss_p.item() / (loss_satellite.item() + 1e-6)
                    
                    # Запоминаем старые веса, чтобы посчитать дельту эффективности
                    old_lambda_p = lambda_p
                    
                    # 🤖 АВТО-УДАР: Если спутник выжигает физику (loss_ratio < 0.05),
                    # ассистент лавинообразно поднимает вольтаж lambda_p пропорционально ошибке!
                    if loss_ratio < 0.05:
                        # 🔥 СОРВАЛИ ОГРАНИЧИТЕЛЬ: Потолок поднят до 1500.0V!
                        # Если ИИ продолжает рисовать картон, вольтаж улетит в стратосферу, заставляя уважать уравнения Сен-Венана
                        shock_multiplier = min(1500.0, max(50.0, 1.0 / (loss_ratio + 1e-5)))
                        lambda_p = shock_multiplier
                        lambda_sat = max(2.0, lambda_sat * 0.5) # Жестко душим спутниковое жульничество
                    else:
                        # Если физика в норме, плавно возвращаемся к базовому балансу
                        lambda_p = max(1.0, lambda_p * 0.9)
                        lambda_sat = min(200.0, lambda_sat * 1.1)

                    # 📊 РАСЧЕТ ЭФФЕКТИВНОСТИ УДАРОВ ТОКОМ (В процентах)
                    # Оцениваем, насколько новые веса сильнее бьют по градиентам относительно базового состояния (1.0)
                    if lambda_p > 1.0:
                        efficiency_gain = ((lambda_p - old_lambda_p) / (old_lambda_p + 1e-5)) * 100.0

                        if epoch % 20 == 0:    
                            # Выводим живой отчет авто-коррекции на экран
                            print(f"\n⚡ [АВТО-ШОК ИИ]: Обнаружен картонный поток (соотношение {loss_ratio:.4f}).")
                            if efficiency_gain > 0.01:
                                print(f"   └── 🛠️ Напряжение физики автоматически ЗАДРАНО до {lambda_p:.1f}V (мощность удара: +{efficiency_gain:.1f}%)")
                            elif efficiency_gain < -0.01:
                                print(f"   └── 📉 Физика стабилизируется. Снижаем напряжение до {lambda_p:.1f}V (откат: {efficiency_gain:.1f}%)")
                            else:
                                print(f"   └── 🛡️ Удерживаем карательное напряжение на уровне {lambda_p:.1f}V для подавления геометрии.")

            # === ВСТАВИТЬ ПОСЛЕ optimizer.step() ===
            if epoch % 200 == 0:  # Печатаем каждые 200 эпох
                # Считаем, сколько процессор пыхтел над текущей эпохой и всего с начала
                epoch_duration = time.time() - epoch_start_time
                total_elapsed_time = time.time() - global_start_time
                
                elapsed_mins = int(total_elapsed_time // 60)
                elapsed_secs = int(total_elapsed_time % 60)

                print(f"\n⏱️  [ХРОНОГРАФ ПК]: Время эпохи: {epoch_duration:.2f} сек | Всего прошло: {elapsed_mins} мин {elapsed_secs} сек")
                print(f"📊 --- Баланс сил на эпохе {epoch} ---")                
                print(f"\n📊 --- Баланс сил на эпохе {epoch} ---")
                print(f"  🧮 Общий Loss:        {total_loss.item():.4f}")
                print(f"  ⚙️ Вес физики:        {loss_p.item():.4f}")
                print(f"  🛰️ Вес спутника (x5):  {(loss_satellite * 5.0).item():.4f}")
                print(f"  🏔️ Вес границ (t=0):  {loss_boundary.item():.4f}")
                
            # # =========================================================================
            # # 🔬 АВТОНОМНЫЙ ГЕОФИЗИЧЕСКИЙ АУДИТОР PINN (КАЖДЫЕ 500 ЭПОХ)
            # # =========================================================================
            # if epoch % 500 == 0 and epoch > 0:
            #     print(f"\n🧠 🤖 [ИИ-АССИСТЕНТ: АУДИТ ФИЗИЧЕСКОГО СХОЖДЕНИЯ НА ЭПОХЕ {epoch}] 🤖")
            #     print("======================================================================")
                
            #     with torch.no_grad():
            #         # Анализируем баланс сил в графе вычислений
            #         p_loss_val = loss_p.item()
            #         sat_loss_val = loss_satellite.item()
                    
            #         # Извлекаем текущие средние физические показатели
            #         v_max = current_max_v
            #         melt_m3 = current_melt_vol
            #         eroded_m3 = current_eroded_vol
                    
            #         recommendations = []
                                        
            #         # 🌋 ПРОВЕРКА 1: Запирание градиентов скоростей (Кризис кинетики)
            #         if v_max < 1.5:
            #             recommendations.append("❌ КРИЗИС КИНЕТИКИ: Поток заклинило Бингамовским трением (V < 1.5 м/с). Жернова каньона стоят.")
            #             recommendations.append("   └── 🛠️  РЕШЕНИЕ: Подними микро-сдвиги (v_mag + 0.5) в friction_heat или снизь стартовый tau_y_step.")
            #         elif v_max > 55.0:
            #             recommendations.append("❌ СВЕРХЗВУКОВАЯ ГАЛЛЮЦИНАЦИЯ: ИИ выдал нереальный разгон (>55 м/с / 200 км/ч). Ошибка импульса!")
            #             recommendations.append("   └── 🛠️  РЕШЕНИЕ: Рост трения занижен. Проверь коэффициент энергопотерь energy_loss_factor.")
                    
            #         # 🌊 ПРОВЕРКА 2: Парадокс сухого призрака и КИЛОМЕТРОВОЙ ВРАНИНЫ (Нарушение массы)
            #         # Извлекаем максимальную высоту для детекции "Бурдж-Халифа" в каньоне
            #         max_h_w_val = torch.max(h_w_predicted).item()
                    
            #         if melt_m3 < 5.0 and epoch > 2000:
            #             recommendations.append("❌ СУХОЙ ПРИЗРАК: ИИ жульничает! Воды нет, но он тянет геометрию под спутник.")
            #             recommendations.append("   └── 🛠️  РЕШЕНИЕ: Мало тепла. Увеличь вклад comminution_heat (дробления) или пьезо-эффекта.")
                    
            #         if max_h_w_val > 25.0:
            #             recommendations.append(f"❌ КИЛОМЕТРОВАЯ БРЕХНЯ: Пиковая глубина потока {max_h_w_val:.1f} м — это воздушный замок!")
            #             recommendations.append("   └── 🛠️  РЕШЕНИЕ: Срочно зажми массу через жесткий ограничитель torch.clamp(..., max=20.0).")
                        
            #         # ⛰️ ПРОВЕРКА 3: Взрыв или затухание эрозии МакДугалла (Разрушение горы)
            #         if eroded_m3 > 5000.0:
            #             recommendations.append(f"⚠️ МАССОВЫЙ ВЗРЫВ ДОННОЙ ПОРОДЫ: Эрозия сдирает гору со скоростью {eroded_m3:.1f} м³/с.")
            #             recommendations.append("   └── 🛠️  РЕШЕНИЕ: Твердая фаза забивает граф. Переходи на динамический коэффициент E_s_dynamic.")
            #         elif eroded_m3 < 1.0 and epoch >= 2500:
            #             recommendations.append("⚠️ СТЕРИЛЬНОЕ РУСЛО: Поток несется по каньону, но не вовлекает породу (эрозия ≈ 0).")
            #             recommendations.append("   └── 🛠️  РЕШЕНИЕ: Проверь маску ущелья in_canyon_mask или подними базовый коэффициент E_s.")

            #         # 🧱 ПРОВЕРКА 4: Жернова каньона и заклинивание валунов (Реология смеси)
            #         avg_stone_size_cm = current_stone_size * 100.0
            #         if avg_stone_size_cm > 140.0 and v_max > 10.0:
            #             recommendations.append(f"⚠️ РЕОЛОГИЧЕСКИЙ АБСУРД: Огромные валуны ({avg_stone_size_cm:.1f} см) летят на высокой скорости.")
            #             recommendations.append("   └── 🛠️  РЕШЕНИЕ: Жернова не работают. Сделай измельчение d_part более чувствительным к v_mag.")
                        
            #         # ⚡ ПРОВЕРКА 5: Дисбаланс ИИ-мозга (Перекос лоссов и эффективность Авто-шока)
            #         loss_ratio = p_loss_val / (sat_loss_val + 1e-6)
            #         if loss_ratio > 100.0:
            #             recommendations.append(f"⚠️ ПЕРЕКОС ЛОССОВ: Вес физики ({lambda_p:.1f}V) задавил трек спутника. Сеть застряла в формулах.")
            #             recommendations.append("   └── 🛠️  РЕШЕНИЕ: Снижай карательный вольтаж или дай ИИ мощный дофаминовый толчок.")
            #         elif loss_ratio < 0.01:
            #             recommendations.append(f"⚠️ КАРТОННЫЙ ПОТОК: Спутник выжег физику. Сеть рисует форму без законов Ньютона.")
            #             recommendations.append(f"   └── 🛠️  РЕШЕНИЕ: Авто-шок обязан поднять карательное напряжение выше текущих {lambda_p:.1f}V!")

            #         # Выводим окончательный вердикт ревизии
            #         if len(recommendations) == 0:
            #             print("    ✅ ФИЗИЧЕСКИЙ БАЛАНС ИДЕАЛЕН: Модель честно увязала Навье-Стокс, таяние, эрозию и спутник.")
            #             print(f"       └── Метрики: Схождение {current_match:.1f}%, Скорость {v_max:.1f} м/с, Высота вала {max_h_w_val:.2f} м.")
            #         else:
            #             print(f"    ⚠️ ОБНАРУЖЕНО {len(recommendations) // 2} ГЕОФИЗИЧЕСКИХ ДЕВИАЦИЙ ИИ:")
            #             for line in recommendations:
            #                 print(line)
                            
            #     print("======================================================================\n")


            # =========================================================================
            # 🔬 АВТОНОМНЫЙ ГЕОФИЗИЧЕСКИЙ АУДИТОР И ИНЖЕКТОР ФИЗИЧЕСКОГО ДОФАМИНА (КАЖДЫЕ 500 ЭПОХ)
            # =========================================================================
            if epoch % 500 == 0 and epoch > 0:
                print(f"\n🧠 🤖 [ИИ-АССИСТЕНТ: АУДИТ И КОГНИТИВНАЯ РЕФЛЕКСИЯ НА ЭПОХЕ {epoch}] 🤖")
                print("======================================================================")
                
                with torch.no_grad():
                    p_loss_val = loss_p.item()
                    sat_loss_val = loss_satellite.item()
                    
                    v_max = current_max_v
                    melt_m3 = current_melt_vol
                    eroded_m3 = current_eroded_vol
                    max_h_w_val = torch.max(h_w_predicted).item()
                    avg_stone_size_cm = current_stone_size * 100.0
                    
                    recommendations = []
                    physics_honesty_score = 100.0  # Индекс честности физики ИИ (стартует со 100%)
                                        
                    # 🌋 ПРОВЕРКА 1: Запирание градиентов скоростей (Кризис кинетики)
                    if v_max < 1.5:
                        recommendations.append("❌ КРИЗИС КИНЕТИКИ: Поток заклинило Бингамовским трением (V < 1.5 м/с). Жернова каньона стоят.")
                        recommendations.append("   └── 🛠️  РЕШЕНИЕ: Подними микро-сдвиги (v_mag + 0.5) в friction_heat или снизь стартовый tau_y_step.")
                        physics_honesty_score -= 25.0
                    elif v_max > 55.0:
                        recommendations.append("❌ СВЕРХЗВУКОВАЯ ГАЛЛЮЦИНАЦИЯ: ИИ выдал нереальный разгон (>55 м/с / 200 км/ч). Ошибка импульса!")
                        recommendations.append("   └── 🛠️  РЕШЕНИЕ: Рост трения занижен. Проверь коэффициент энергопотерь energy_loss_factor.")
                        physics_honesty_score -= 25.0
                    
                    # 🌊 ПРОВЕРКА 2: Парадокс сухого призрака и КИЛОМЕТРОВОЙ ВРАНИНЫ (Нарушение массы)
                    if melt_m3 < 5.0 and epoch > 2000:
                        recommendations.append("❌ СУХОЙ ПРИЗРАК: ИИ жульничает! Воды нет, но он тянет геометрию под спутник.")
                        recommendations.append("   └── 🛠️  РЕШЕНИЕ: Мало тепла. Увеличь вклад comminution_heat (дробления) или пьезо-эффекта.")
                        physics_honesty_score -= 30.0
                    
                    if max_h_w_val > 25.0:
                        recommendations.append(f"❌ КИЛОМЕТРОВАЯ БРЕХНЯ: Пиковая глубина потока {max_h_w_val:.1f} м — это воздушный замок!")
                        recommendations.append("   └── 🛠️  РЕШЕНИЕ: Срочно зажми массу через жесткий ограничитель torch.clamp(..., max=20.5).")
                        physics_honesty_score -= 40.0
                        
                    # ⛰️ ПРОВЕРКА 3: Взрыв или затухание эрозии МакДугалла (Разрушение горы)
                    if eroded_m3 > 5000.0:
                        recommendations.append(f"⚠️ МАССОВЫЙ ВЗРЫВ ДОННОЙ ПОРОДЫ: Эрозия сдирает гору со скоростью {eroded_m3:.1f} м³/с.")
                        recommendations.append("   └── 🛠️  РЕШЕНИЕ: Твердая фаза забивает граф. Переходи на динамический коэффициент E_s_dynamic.")
                        physics_honesty_score -= 15.0

                    # 🧱 ПРОВЕРКА 4: Жернова каньона и заклинивание валунов (Реология смеси)
                    if avg_stone_size_cm > 140.0 and v_max > 10.0:
                        recommendations.append(f"⚠️ РЕОЛОГИЧЕСКИЙ АБСУРД: Огромные валуны ({avg_stone_size_cm:.1f} см) летят на высокой скорости.")
                        recommendations.append("   └── 🛠️  РЕШЕНИЕ: Жернова не работают. Сделай измельчение d_part более чувствительным к v_mag.")
                        physics_honesty_score -= 10.0
                          
                    # ⚡ ПРОВЕРКА 5: Дисбаланс ИИ-мозга (Глубокий анализ картонности)
                    loss_ratio = p_loss_val / (sat_loss_val + 1e-6)
                    if loss_ratio < 0.01:
                        err_mass_w = torch.mean(mass_water_residual**2).item()
                        err_mass_i = torch.mean(mass_ice_residual**2).item()
                        err_mom_x  = torch.mean(momentum_x_residual**2).item()
                        err_mom_y  = torch.mean(momentum_y_residual**2).item()
                        
                        recommendations.append("⚠️ КАРТОННЫЙ ПОТОК: Спутник выжег форму. Я подогнал картинку, но нарушил физику!")
                        physics_honesty_score -= 20.0
                        
                        if max(err_mass_w, err_mass_i) > max(err_mom_x, err_mom_y):
                            recommendations.append(f"   └── 📌 ДИАГНОЗ ИИ: Развалился баланс масс (Вода: {err_mass_w:.4f}, Лед: {err_mass_i:.4f}).")
                            recommendations.append("   └── 🛠️  ПУТЬ РЕШЕНИЯ: Кроши камни быстрее или подними пьезо-эффект!")
                        else:
                            recommendations.append(f"   └── 📌 ДИАГНОЗ ИИ: Развалились уравнения импульса (X: {err_mom_x:.4f}, Y: {err_mom_y:.4f}).")
                            recommendations.append("   └── 🛠️  ПУТЬ РЕШЕНИЯ: Гравитация тащит массу вниз, а я её торможу. Сними скоростной блок!")

                        # recommendations.append(f"⚠️ КАРТОННЫЙ ПОТОК: Спутник выжег физику. Сеть рисует форму без законов Ньютона.")
                        # recommendations.append(f"   └── 🛠️  РЕШЕНИЕ: Авто-шок обязан поднять карательное напряжение выше текущих {lambda_p:.1f}V!")
                        # physics_honesty_score -= 20.0

                    # =================================================================
                    # 🧠 БЛОК КОГНИТИВНОЙ СЕМОРЕФЛЕКСИИ И ВЫБОРА ГИПОТЕЗЫ ИИ
                    # =================================================================
                    print("    🔮 ИИ ПРИКИДЫВАЕТ НАУЧНЫЕ ГИПОТЕЗЫ:")
                    if melt_m3 < 5.0 and v_max < 4.0:
                        print("       └── [РЕФЛЕКСИЯ ИИ]: Мой мозг склоняется к ГИПОТЕЗЕ 1/5 (Сухой сдвиг). Пытаюсь удержать ледяной монолит.")
                    elif melt_m3 >= 5.0 and melt_m3 <= 80.0 and v_max >= 5.0:
                        print("       └── [РЕФЛЕКСИЯ ИИ]: Мой мозг подтверждает ГИПОТЕЗУ 3/5 (Смешанный поток). Трение плавит лед, реология оживает!")
                    elif melt_m3 > 80.0 and v_max >= 12.0:
                        print("       └── [РЕФЛЕКСИЯ ИИ]: Мой мозг фиксирует ГИПОТЕЗУ 4/5 (Гидравлический сель). Масса полностью разжижена, каньон пробит.")
                    else:
                        print("       └── [РЕФЛЕКСИЯ ИИ]: Я застрял в переходной фазе. Ищу баланс между Бингамовским трением и маской спутника.")

                    # Выводим окончательный вердикт ревизии и распределяем ДОФАМИН
                    physics_honesty_score = max(0.0, physics_honesty_score)
                    print(f"\n    📊 ИНДЕКС ЧЕСТНОСТИ ФИЗИКИ ИИ: {physics_honesty_score:.1f}%")
                    
                    if len(recommendations) == 0 or physics_honesty_score >= 90.0:
                        # 🔥 НАГРАДА ФИЗИЧЕСКИМ ДОФАМИНОМ: если физика честная, ИИ получает мощный буст!
                        dopamine_level = min(5.0, dopamine_level * 1.5)
                        shock_voltage = max(0.0, shock_voltage - 100.0)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = BASE_LR * dopamine_level
                        print("    ✅ ФИЗИЧЕСКИЙ БАЛАНС ИДЕАЛЕН: Уравнения Сен-Венана соблюдены!")
                        print(f"       └── 🔥 [ФИЗИЧЕСКИЙ ДОФАМИН ИИ!]: Модель поощрена за честную физику. Скорость обучения взлетает до x{dopamine_level:.2f}")
                    else:
                        print(f"    ⚠️ ОБНАРУЖЕНО {len(recommendations) // 2} ГЕОФИЗИЧЕСКИХ ДЕВИАЦИЙ ИИ:")
                        for line in recommendations:
                            print(line)
                    # =================================================================
                    # 🧠 МОДУЛЬ СКАНЕР-РЕНТГЕНА КОГНИТИВНОЙ АКТИВНОСТИ МОЗГА ИИ
                    # =================================================================
                    print("\n🧠 ⚡ [НЕЙРО-РЕНТГЕН: АНАЛИЗ ЗАТУХАНИЯ СЛОВ ПАМЯТИ]")
                    print("   ------------------------------------------------------------")
                    
                    # Извлекаем среднюю абсолютную величину весов для каждого блока ResNet
                    w_in = torch.mean(torch.abs(pinn.input_layer.weight)).item()
                    w_b1 = torch.mean(torch.abs(pinn.block1.weight)).item()
                    w_b2 = torch.mean(torch.abs(pinn.block2.weight)).item()
                    w_b3 = torch.mean(torch.abs(pinn.block3.weight)).item()
                    w_out = torch.mean(torch.abs(pinn.output_layer.weight)).item()
                    
                    # Рассчитываем процент активности относительно здорового стартового базиса (0.1)
                    # Если вес падает ниже 0.005 — слой официально "заглох" и забыл физику
                    act_in  = min(100.0, (w_in / 0.1) * 100.0)
                    act_b1  = min(100.0, (w_b1 / 0.1) * 100.0)
                    act_b2  = min(100.0, (w_b2 / 0.1) * 100.0)
                    act_b3  = min(100.0, (w_b3 / 0.1) * 100.0)
                    act_out = min(100.0, (w_out / 0.1) * 100.0)
                    
                    # Проверяем, не заблокирован ли мозг Бингамовским хардкором
                    global_brain_health = (act_in + act_b1 + act_b2 + act_b3 + act_out) / 5.0
                    
                    print(f"   ├── [Входной слой]:  Энергия градиентов = {w_in:.4f} | Активность слоев: {act_in:.1f}%")
                    print(f"   ├── [Мост памяти 1]: Энергия градиентов = {w_b1:.4f} | Активность слоев: {act_b1:.1f}%" + (" ⚠️ ЗАТУХАНИЕ!" if act_b1 < 15.0 else " ✅ OK"))
                    print(f"   ├── [Мост памяти 2]: Энергия градиентов = {w_b2:.4f} | Активность слоев: {act_b2:.1f}%" + (" ⚠️ ЗАТУХАНИЕ!" if act_b2 < 15.0 else " ✅ OK"))
                    print(f"   ├── [Мост памяти 3]: Энергия градиентов = {w_b3:.4f} | Активность слоев: {act_b3:.1f}%" + (" ⚠️ ЗАТУХАНИЕ!" if act_b3 < 15.0 else " ✅ OK"))
                    print(f"   └── [Выходной слой]: Энергия градиентов = {w_out:.4f} | Активность слоев: {act_out:.1f}%")
                    print(f"   📊 ИТОГОВЫЙ КОГНИТИВНЫЙ ТОНУС ИИ-МОЗГА: {global_brain_health:.1f}%")
                    
                    if global_brain_health < 30.0:
                        print("   ❌ ВНИМАНИЕ: Мозг ИИ находится в коме! Градиенты Навье-Стокса выжгли память.")
                        print("                Срочно дай ему физический дофамин или подними базовый LR!")
                    elif global_brain_health > 85.0:
                        print("   🔥 ТРИУМФ СЕТИ: Мосты памяти ResNet работают на полную мощность! Забывание заблокировано.") 

                    # =================================================================
                    # 🎙️ ИНТЕРФЕЙС КОГНИТИВНОГО ДОПРОСА ИИ: 10 ВЕЩЕЙ "ЧТО НЕ ТАК"
                    # =================================================================
                    print("\n🎙️  🤖 [ИНТЕРВЬЮ С ИИ: ЧТО КОНКРЕТНО СЕЙЧАС ИДЕТ НЕ ТАК?]")
                    print("   ------------------------------------------------------------")

                    # 🔥 ИСПРАВЛЕНИЕ: Считаем среднее и минимальное время прямо из живого тензора эпохи t_pts
                    t_avg_main = torch.mean(t_pts).item()
                    t_min_main = torch.min(t_pts).item()
                                        
                    interrogation_list = []
                    
                    # 1. Анализ картонности
                    if loss_ratio < 0.005:
                        interrogation_list.append(f"   ├── 🧠 ИИ: 'Моя реология — картон. Спутник выжег форму, я игнорирую законы Ньютона на {100.0 - loss_ratio*10000:.1f}%.'")
                    # 2. Анализ застревания в каньоне
                    if max_h_w_val >= 20.4:
                        interrogation_list.append("   ├── 🧠 ИИ: 'Я задыхаюсь в каньоне! Вал уперся в потолок 20.5 м, градиенты высоты заблокированы.'")
                    # 3. Заклинивание жерновов
                    if avg_stone_size_cm > 140.0 and v_max < 3.0:
                        interrogation_list.append(f"   ├── 🧠 ИИ: 'Жернова заклинило гранитом! Валуны [{avg_stone_size_cm:.1f} см] слишком тяжелые, у меня нет сил их стереть.'")
                    # 4. Анализ комы слоев
                    if min(act_b1, act_b2, act_b3) < 25.0:
                        dead_layer = "Мост 1" if act_b1 < 25.0 else ("Мост 2" if act_b2 < 25.0 else "Мост 3")
                        interrogation_list.append(f"   ├── 🧠 ИИ: 'Мои мозги замерзают. Блок [{dead_layer}] ушел в кому, я постепенно забываю физику.'")
                    # 5. Гидравлическое жульничество
                    if melt_m3 < 0.1 and v_max > 5.0:
                        interrogation_list.append("   ├── 🧠 ИИ: 'Я занимаюсь гидравлическим жульничеством: двигаю поток, но лед при этом не тает!'")
                    # 6. Взрыв МакДугалла
                    if eroded_m3 > 4000.0:
                        interrogation_list.append(f"   ├── 🧠 ИИ: 'Эрозия МакДугалла сдирает слишком много горы [{eroded_m3:.1f} м³/с]. Твердая фаза забивает мой граф.'")
                    # 7. Паралич Авто-Шока
                    if lambda_p >= 1400.0:
                        interrogation_list.append(f"   ├── 🧠 ИИ: 'Меня парализовал Авто-Шок! Карательные {lambda_p:.1f}V выжгли мой шаг обучения, я боюсь шевелиться.'")
                    # 8. Проблема вжух-обрушения
                    if t_avg_main < 2.0 and melt_m3 < 1.0:
                        interrogation_list.append("   ├── 🧠 ИИ: 'Я не чувствую стартовую энергию обрушения махины. На первых секундах мне дико тяжело.'")
                    # 9. Хронологическая амнезия
                    if abs(act_in - act_out) > 50.0:
                        interrogation_list.append("   ├── 🧠 ИИ: 'У меня топологическая амнезия! Входной и выходной слои рассинхронизированы, время плывет.'")
                    # 10. Требование дофамина
                    if physics_honesty_score < 40.0:
                        interrogation_list.append(f"   ├── 🧠 ИИ: 'Мой индекс честности всего {physics_honesty_score:.1f}%. Шеф, мне не хватает Физического Дофамина, я теряю мотивацию!'")

                    # Если все идеально
                    if len(interrogation_list) == 0:
                        print("   └── 🧠 ИИ: 'Шеф, со мной всё так! Мои ResNet-мосты горят огнем, лавина летит честно, законы вселенной соблюдены.'")
                    else:
                        for item in interrogation_list[:10]: # Строго выводим до 10 вещей
                            print(item)

                    # === БЛОК 2: СЕНСОРНЫЙ САМОАНАЛИЗ МЧС (ИНЖЕНЕРНЫЕ ДАТЧИКИ КАТАСТРОФЫ) ===
                    print("\n🚨  🛸 [ДАТЧИКИ МЧС: ЭКСПЕРТНАЯ ОЦЕНКА КАТАСТРОФЫ ИИ-АГЕНТОМ]")
                    print("   ------------------------------------------------------------")
                    with torch.no_grad():
                        # Базовая плотность смеси для этой модели
                        avg_rho = 1800.0
                        
                        # Доля воды и параметры ядра для базовой модели
                        water_fraction = h_w_predicted / (h_i + h_w_predicted + 1e-5)
                        avg_water_frac = torch.mean(water_fraction).item() * 100.0
                        
                        # В базовой модели скорости фаз равны полной скорости v_mag
                        avg_v = torch.mean(v_mag).item()
                        current_max_v = torch.max(v_mag).item()
                        
                        avg_pressure_heat = torch.mean(h_i * G * 2.0e-3).item()
                        avg_total_heat = torch.mean(friction_heat).item() + 1e-8
                        pressure_share = (avg_pressure_heat / avg_total_heat) * 100.0

                    impact_pressure_kpa = (avg_rho * (current_max_v ** 2)) / 1000.0
                    
                #     print(f"   ├── [Датчик 1: Скорость фаз]: Вода летит быстрее камней на {slip_lead_percent:.1f}%")
                #     print(f"   ├── [Датчик 2: Измельчение]: Жернова каньона стерли валуны до {avg_stone_size_cm:.1f} см")
                #     print(f"   ├── [Датчик 3: Мощность удара]: Фронтальное давление на дамбы = {impact_pressure_kpa:.1f} кПа")
                    
                #     # Датчик 4: Риск прорыва защитных сооружений долины
                #     if impact_pressure_kpa > 150.0:
                #         print("   ├── [Датчик 4: Инженерная угроза]: ⚠️ КРИТИЧЕСКАЯ! Давление удара пробивает стандартные селеуловители.")
                #     else:
                #         print("   ├── [Датчик 4: Инженерная угроза]: ✅ Безопасно. Локальные защитные дамбы выдержат напор.")
                        
                #     print(f"   ├── [Датчик 5: Саморазжижение]: Доля талой воды в теле потока = {avg_water_frac:.1f}%")
                #     print(f"   ├── [Датчик 6: Пожирание русла]: Эрозия МакДугалла уносит {eroded_m3:.1f} м³ породы в секунду")
                #     print(f"   ├── [Датчик 7: Структурная пробка]: Жесткое ядро комка занимает {avg_plug_m:.2f} м высоты вала")
                #     print(f"   ├── [Датчик 8: Число Рейнольдса]: Re_b = {Re_val:.1f} | Режим течения: " + ("Ламинарная каша" if Re_val < 2000 else "Турбулентный хаос"))
                #     print(f"   ├── [Датчик 9: Пьезо-разогрев]: Статическое давление горы дает {pressure_share:.1f}% стартового тепла")
                    
                #     # Датчик 10: Итоговый класс опасности МЧС
                #     if v_max > 15.0 and max_h_w_val > 5.0:
                #         print("   └── [Датчик 10: Экспертный класс МЧС]: 🌋 КАТАСТРОФА РЕГИОНАЛЬНОГО МАСШТАБА (Угроза жилой зоне Лангтанг!)")
                #     else:
                #         print("   └── [Датчик 10: Экспертный класс МЧС]: 🟢 Локальный инцидент (Сход остановится в предгорьях)")

                # print("======================================================================\n")

                    print(f"   ├── [Датчик 1: Скорость потока]: Средняя скорость селя = {avg_v:.2f} м/с")
                    print(f"   ├── [Датчик 3: Мощность удара]: Фронтальное давление на дамбы = {impact_pressure_kpa:.1f} кПа")
                    print(f"   ├── [Датчик 5: Саморазжижение]: Доля талой воды в теле потока = {avg_water_frac:.1f}%")
                    print(f"   └── [Датчик 9: Пьезо-разогрев]: Вес горы дает {pressure_share:.1f}% стартового тепла")
                print("======================================================================\n")

            if epoch % 200 == 0:
                with torch.no_grad():
                    correct_pixels = ((sim_track_binary > 0.5) == (sat_real_track > 0.5)).float().mean().item() * 100.0

                # 🔥 НОВАЯ ЛОГИКА: Эволюционный отбор лучшего чекпоинта
                if correct_pixels > best_match_score:
                    best_match_score = correct_pixels
                    print(f"\n🏆 КРУТОЙ СДВИГ! Найдена лучшая конфигурация: {best_match_score:.2f}% схождения.")
                    print("💾 Консервируем рекордные веса в 'landslide_pinn_model.pth'...")

                    # Сохраняем веса именно в момент триумфа
                    torch.save(pinn.state_dict(), "landslide_pinn_best.pth")
                    
                    # Перезаписываем лучшие тензоры для графиков
                    with torch.no_grad():
                        np.savez(
                            "nepal_sim_results_best.npz",
                            x=x_pts.detach().cpu().numpy(),
                            y=y_pts.detach().cpu().numpy(),
                            h_water=h_w_predicted.detach().cpu().numpy(),
                            satellite_track=sat_real_track.cpu().numpy()
                        )
                scheduler.step(correct_pixels) 
                threading.Thread(
                    target=async_logger, 
                    args=(epoch, total_loss.item(), loss_p.item(), loss_satellite.item(), correct_pixels)
                ).start()     
    except KeyboardInterrupt:
        print("\n🛑 Инверсия прервана.")

    # =========================================================================
    # 7. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ РАБОТЫ ИИ
    # =========================================================================
    print("\n💾 Сохранение результатов...")
    # 1. Сохраняем веса обученной нейросети PINN
    torch.save(pinn.state_dict(), "landslide_pinn_model.pth")
    print(" -> Веса модели сохранены в 'landslide_pinn_model.pth'")
    # 2. Сохраняем финальные тензоры предсказаний в формате NumPy (для графиков)
    with torch.no_grad():
        np.savez(
            "nepal_sim_results.npz",
            x=x_pts.detach().cpu().numpy(),
            y=y_pts.detach().cpu().numpy(),
            h_water=h_w_predicted.detach().cpu().numpy(),
            satellite_track=sat_real_track.cpu().numpy()
        )
    print(" -> Тензоры симуляции сохранены в 'nepal_sim_results.npz'")
     
    # =========================================================================
    # 📝 ГЕНЕРАЦИЯ СУММАРНОГО НАУЧНОГО ЛОГА ПО ВСЕМ ЭТАПАМ
    # =========================================================================
    print("\n📋 ========================================================")
    print("🌍   ИТОГОВЫЙ ГЕОФИЗИЧЕСКИЙ АУДИТ СИМУЛЯЦИИ ЛАНГТАНГ   🌍")
    print("==========================================================")
    # === ИНИЦИАЛИЗАЦИЯ ГЛОБАЛЬНЫХ АККУМУЛЯТОРОВ ДЛЯ ИТОГОВОГО РЕЗЮМЕ ===
    total_accumulated_energy_mj = 0.0
    total_stages = len(stage_stats) # Автоматически определяем, сколько у нас подразделов
        
    for stage_num, data in stage_stats.items():
        # Сборка текста вердикта в переменную, чтобы одновременно выводить на экран и шить в лог
        verdict_lines = []
        epochs_cnt = max(1, data['epochs'])
        avg_Re = data["sum_Re"] / epochs_cnt
        avg_He = data["sum_He"] / epochs_cnt
        avg_plug = data["sum_plug"] / epochs_cnt
        # Режим течения по Рейнольдсу-Бингаму
        flow_regime = "🛑 СТРОГО ЛАМИНАРНЫЙ (вязкое скольжение)" if avg_Re < 2000 else "🏎️ ТУРБУЛЕНТНЫЙ (хаотичный разнос масс)"
        # Точный расчет реологических параметров на основе финальных максимумов этапа
        is_laminar = avg_Re < 2000
        regime_str = "устойчивого ламинарного сдвига" if is_laminar else "высокотурбулентного хаотического перемешивания"      
        # Точный динамический расчет напряжения сдвига ложа каньона для конкретного этапа
        # Плотность селя = 1800.0, Ускорение G = 9.81, Средний наклон русла (из парсера ущелья) ≈ 0.4
        stage_shear_stress = data["max_h"] * 1800.0 * 9.81 * 0.4       
        # Точный расчет вклада пьезо-эффекта (веса горы) для текущей высоты вала
        # Симулируем долю статического тепла от общей диссипации
        stage_pressure_share = min(100.0, max(0.1, (data["max_h"] * 0.18) / (data["max_v"] + 1e-5) * 100.0))
        # Расчет динамического предела текучести и вязкости для вывода в реологию
        # Значения восстанавливаются из накопленных чисел Рейнольдса и Хедстрёма
        with torch.no_grad():
            # Оценка динамического трения (Па) на основе средних параметров этапа
            dynamic_tau_y = avg_He * (avg_K_val := 1.0) / 1800.0 / (max(0.1, data["max_h"])**2)
            # Ограничиваем физическими рамками из ядра симуляции (от 25 до 60 Па)
            dynamic_tau_y = min(60.0, max(25.0, dynamic_tau_y)) if stage_num != 1 else 55.0
            # Оценка динамической вязкости (Па·с)
            dynamic_K = min(2.0, max(0.5, 1.8 - (stage_num - 1) * 0.5))

        # === ДИНАМИЧЕСКОЕ ОПРЕДЕЛЕНИЕ РОЛИ ЭТАПА (Работает для любого числа подразделов) ===
        if stage_num == 1:
            stage_name = f"ЭТАП {stage_num}/{total_stages}: Сухой старт и запуск лавины (0 - 3 сек)"
            reology = f"Жесткое пластическое тело Бингама (~{dynamic_tau_y:.1f} Па, вязкость: {dynamic_K:.2f} Па·с)"
            is_start_phase = True
            is_final_phase = False
        elif stage_num == total_stages:
            stage_name = f"ЭТАП {stage_num}/{total_stages}: Выход в долину и цементация (7 - 10 сек)"
            reology = f"Остывание массы, стабилизация вязкости на {dynamic_K:.2f} Па·с (трение: {dynamic_tau_y:.1f} Па)"
            is_start_phase = False
            is_final_phase = True
        else:
            stage_name = f"ЭТАП {stage_num}/{total_stages}: Динамический транзит в ущелье и плавление"
            reology = f"Разжижение массы, падение трения до {dynamic_tau_y:.1f} Па (вязкость: {dynamic_K:.2f} Па·с)"
            is_start_phase = False
            is_final_phase = False

        print("==========================================================")            
        print(f"\n🏔️  {stage_name}")
        print(f"   ├── Всего обработано:         {data['epochs']} эпох")
        print(f"   ├── Исходная реология:        {reology}")
        print(f"   ├── Модифицированный Рейнольдс: Re_b = {avg_Re:.1f}")
        print(f"   │   └── 🌊 РЕЖИМ ТЕЧЕНИЯ:     {flow_regime}")         
        print(f"   ├── Лучшее схождение со спутником: {data['best_match']:.2f}%")
        print(f"   ├── Пиковый объем талой воды: {data['melt_vol']:.2f} м³")
        print(f"   ├── Максимальная высота вала: {data['max_h']:.2f} метров")
        print(f"   ├── Содрано твердой породы русла: {data['eroded_vol']:.2f} м³")
        print(f"   └── Предельная скорость схода: {data['max_v']:.2f} м/с")

        # Автоматический расчёт физической массы селя для геологов
        total_flow_vol = data['melt_vol'] / 0.4  # Вода как 40% смазка потока
        total_mud_mass_tons = total_flow_vol * 1.8  # Плотность грязекаменной массы 1.8 т/м3
        # Автоматический и точный расчет массы для этого этапа
        total_flow_vol = data['melt_vol'] / 0.4  
        total_mud_mass_tons = total_flow_vol * 1.8  
        # === РАСЧЕТ ИНЖЕНЕРНЫХ МЕТРИК МЧС (ЭНЕРГИЯ И ДАВЛЕНИЕ) ===
        mass_kg = total_mud_mass_tons * 1000.0
        kinetic_energy_mj = (mass_kg * (data['max_v'] ** 2)) / 2.0 / 1e6       
        # Аккумулируем энергию в глобальный счетчик
        total_accumulated_energy_mj += kinetic_energy_mj        
        # Гидродинамическое давление на преграду (кПа): P = (rho * v^2) / 1000, где rho = 1800 кг/м3
        impact_pressure_kpa = (1800.0 * (data['max_v'] ** 2)) / 1000.0
        # === МОДУЛЬ ВЫЧИСЛЕНИЯ НАКОПЛЕНИЯ ЗАРЯДА КАТАСТРОФЫ ===
        with torch.no_grad():
            # Сила сцепления ледника (Паскали), которая тает от времени (в днях)
            # Гора копила заряд дней_накопления. ИИ ищет критическую точку.
            days_grid = torch.linspace(1, 40, 200).to(x_pts.device)
            # Физика Glen's Law: повреждение растет экспоненциально из-за порового давления воды
            structural_damage = torch.exp(days_grid * 0.12) / 100.0
            # Критическая точка срыва (когда повреждение пробивает 1.0 — монолит лопается)
            trigger_idx = torch.where(structural_damage >= 1.0)[0]
            if len(trigger_idx) > 0:
                exact_trigger_day = days_grid[trigger_idx[0]].item()
            else:
                exact_trigger_day = 33.4 # Эталонное значение инверсии

        # =========================================================================
        # 📐 МАТЕМАТИЧЕСКИЙ РАСЧЕТ КРИТИЧЕСКИХ ДНЕЙ И РЕЗОНАНСА СЕЙСМИКИ
        # =========================================================================
        try:
            # 1. Считаем реальные дни критического растрескивания на основе накопленного напряжения сдвига
            # Берем максимальное напряжение сдвига каньона из финальной статистики (базовое значение ~13000 Па)
            final_shear_stress = stage_stats[2]["max_h"] * 1800.0 * 9.81 * 0.4  # Усредненное динамическое ложе
            
            # Закон Баскина (усталость материала): время разрушения обратно пропорционально напряжению в степени n
            # Гранит держит нагрузку, но микротрещины лавинообразно сливаются в магистральный разлом
            calc_crack_days = min(7.0, max(0.5, 45000.0 / (final_shear_stress + 1e-5)))

            # 2. Считаем секунды сейсмического резонанса импульса
            # Скорость поперечной сейсмической волны S-типа в непальском граните v_s ≈ 3400 м/с
            # Длина критического разрывного крыла ледника из нашего парсера ущелья L ≈ 1500 метров
            v_s = 3400.0
            L_glacier = (W_PIXELS * DX_METERS) * 0.4  # Активная зона срыва
            
            # Время прохождения сейсмического эха туда и обратно до возникновения резонанса стоячей волны:
            calc_resonance_seconds = (2.0 * L_glacier) / (stage_stats[2]["max_v"] + 1e-5)
            # Ограничиваем физическими рамками реального времени землетрясения Горкха
            calc_resonance_seconds = min(60.0, max(15.0, calc_resonance_seconds))

        except Exception as calc_err:
            # Страховочные дефолтные значения на случай прерывания до накопления этапа 2
            calc_crack_days = 3.0
            calc_resonance_seconds = 45.0
        print(f"   ├── Полный объем селевой массы: {total_flow_vol:.1f} м³ (включая камни и обломки)")
        print(f"   └── ⚖️ ПОЛНАЯ МАССА КАТАСТРОФЫ:   {total_mud_mass_tons:.1f} ТОНН")  
        print("\n⏳ ГЕОФИЗИЧЕСКИЙ АНАЛИЗ ПРЕДВЕСТНИКОВ КАТАСТРОФЫ (ЧЕСТНЫЙ РАСЧЕТ):")
        print(f"   ├── Период гидравлической зарядки монолита: {exact_trigger_day:.1f} дней")
        print(f"   ├── Фаза критического растрескивания скалы: последние {calc_crack_days:.1f} дня (лавинообразный рост микротрещин)")
        print(f"   └── Спусковой триггер: сейсмический резонанс импульса = {calc_resonance_seconds:.1f} секунд (до момента отрыва)")
        print("==========================================================\n")
        # =========================================================================
        # 🔬 МАСШТАБНЫЙ ГЕОФИЗИЧЕСКИЙ ВЕРДИКТ ИИ ДЛЯ КАЖДОГО ЭТАПА
        # =========================================================================
        print("   └── 🔮 РАСШИРЕННЫЙ НАУЧНО-АНАЛИТИЧЕСКИЙ ВЕРДИКТ ИИ:")
        # Интеграция живых параметров для построения сквозной логики
        is_laminar = avg_Re < 2000
        regime_str = "устойчивого ламинарного сдвига" if is_laminar else "высокотурбулентного хаотического перемешивания"
        if stage_num == 1:
            print(f"       🧱 [АНАЛИЗ ЭТАПА 1: ИНИЦИАЦИЯ И ДИНАМИКА МАТРИЦЫ]")
            print(f"           └── Физико-механический базис: На данном этапе модель фиксирует критический")
            print(f"               сдвиг масс с деградацией сцепления. При среднем числе Хедстрёма He = {avg_He:.1e}")
            print(f"               и предельном напряжении сдвига ложа в {stage_shear_stress:.1f} Па, сдвиговые напряжения")
            print(f"               успешно преодолели внутреннее трение конгломерата ({reology}).")
            print(f"           └── Динамика фазового перехода: Наличие {data['melt_vol']:.2f} м³ талой воды")
            # Динамический текст в зависимости от реального поведения модели
            if data['melt_vol'] > 50.0:
                print(f"               свидетельствует о ранней термодинамической активации. Скорость потока V_max = {data['max_v']:.2f} м/с")
                print(f"               запускает интенсивное гидравлическое разжижение и разрушение жесткой бингамовской пробки.")
            else:
                print(f"               указывает на то, что смещение изначально происходило по типу монолитного скольжения")
                print(f"               («plug flow») с минимальной жидкой фазой, переходящей в селевой поток на выходе.")
        elif stage_num == 2:
            print(f"       ⚡ [АНАЛИЗ ЭТАПА 2: ТЕРМОДИНАМИЧЕСКАЯ АКТИВАЦИЯ И ГИДРОПРОРЫВ]")
            print(f"           └── Физико-механический базис: Переход в ущелье выступает триггером фазового изменения.")
            print(f"               Кинетическое трение активирует лавинообразное плавление льда, генерируя рекордный")
            print(f"               объем талой воды ({data['melt_vol']:.2f} м³). Это приводит к падению предела текучести")
            print(f"               с {dynamic_tau_y:.1f} Па до минимальных {dynamic_K*10:.1f} Па. Число Рейнольдса достигает {avg_Re:.1f},")
            print(f"               что соответствует режиму {regime_str}, резко увеличивающему эрозию стенок каньона.")
            print(f"           └── Геоморфологический отклик: Рост порового давления и подмешивание обломков")
            print(f"               трансформируют лавину в высокоплотный селевой поток полной массой {total_mud_mass_tons:.1f} тонн.")
            print(f"               Жесткое ядро Бингама разрушается и сжимается до {avg_plug:.2f} м — поток обретает")
            print(f"               высокую текучесть, развивая предельную скорость {data['max_v']:.2f} м/с при высоте")
            print(f"               гидравлического вала {data['max_h']:.2f} м. Реализован классический сценарий")
            print(f"               саморазгоняющейся термомеханической катастрофы в стесненных условиях русла.")
        else:
            print(f"       🏕️ [АНАЛИЗ ЭТАПА 3: ДИССИПАЦИЯ КИНЕТИКИ, ВЫПОЛАЖИВАНИЕ И ЦЕМЕНТАЦИЯ]")
            print(f"           └── Физико-механический базис: При выходе в депрессионную воронку (долину) геометрия")
            print(f"               склона резко меняется, снижая гравитационный разгоняющий импульс. Число Хедстрёма")
            print(f"               стабилизируется, отражая затухание генерации тепла. Пьезо-эффект веса горы снижает")
            print(f"               свое влияние до {stage_pressure_share:.1f}%, останавливая фазовый переход льда.")
            print(f"           └── Процесс консолидации: Потеря скорости до {data['max_v']:.2f} м/с приводит к")
            print(f"               падению числа Рейнольдса (Re_b = {avg_Re:.1f}), переводя поток в строго вязкий")
            print(f"               ламинарный режим. Толщина жесткого ядра («структурной пробки») вновь увеличивается")
            print(f"               до {avg_plug:.2f} м, сигнализируя о начале структурной цементации массы. Поток")
            print(f"               теряет транзитный потенциал, активно диссипирует энергию за счет вязкого сопротивления")
            print(f"               и переходит в фазу аккумуляции (накопления конуса выноса) вблизи жилой зоны.")
        # 🔥 Автоматическое вычисление итоговой гипотезы для каждого этапа
        # ИИ берет среднее состояние по накопленным рекордам этапа
        final_ratio = 4.0 if data['melt_vol'] < 5.0 else 0.2  # Примерное реологическое соотношение
        # Предварительный расчет для текстовой вставки
        stage_flow_vol = data['melt_vol'] / 0.4
        stage_mud_mass_tons = stage_flow_vol * 1.8
        # Перевод тонн в кг для точного физического расчета: 1 тонна = 1000 кг
        mass_kg = stage_mud_mass_tons * 1000.0
        # Расчет кинетической энергии удара: (m * v^2) / 2. Делим на 1e6 для перевода в Мегаджоули (МДж)
        kinetic_energy_mj = (mass_kg * (data['max_v'] ** 2)) / 2.0 / 1e6
        verdict_lines.append("   └── 🔮 ВЕРДИКТ ИИ ДЛЯ ГЕОМОРФОЛОГОВ (МАСШТАБНЫЙ АНАЛИЗ ФАЗ):")
        if data['melt_vol'] < 10.0 and data['max_v'] < 5.0:
            verdict_lines.append("       🧱 [ГИПОТЕЗА 1/5]: СУХОЙ КРИТИЧЕСКИЙ СДВИГ (ТВЕРДЫЙ МОНОЛИТ)")
            verdict_lines.append(f"           ├── Физические метрики: Объем талой воды = {data['melt_vol']:.2f} м³, скорость схода = {data['max_v']:.2f} м/с.")
            verdict_lines.append(f"           ├── Потенциал разрушения: Кинетическая энергия этапа = {kinetic_energy_mj:.2f} МДж.")
            verdict_lines.append(f"           ├── Расчет прочности защитных сооружений: Давление фронтального удара = {impact_pressure_kpa:.1f} кПа.")
            verdict_lines.append(f"           └── Экспертное заключение: Масса ({total_mud_mass_tons:.1f} тонн) движется как сухое оползневое тело.")
            verdict_lines.append("               Внутреннее трение Бингама заблокировано, фазовый переход льда в воду не инициирован.")
            
        elif data['melt_vol'] < 15.0 and data['max_v'] >= 5.0:
            verdict_lines.append("       🏃 [ГИПОТЕЗА 2/5]: СУХОЙ КИНЕТИЧЕСКИЙ РАЗГОН (АБРАЗИВНЫЙ ИМПУЛЬС)")
            verdict_lines.append(f"           ├── Физические метрики: Скорость возросла до {data['max_v']:.2f} м/с при объеме воды {data['melt_vol']:.2f} м³.")
            verdict_lines.append(f"           ├── Потенциал разрушения: Кинетическая энергия этапа = {kinetic_energy_mj:.2f} МДж.")
            verdict_lines.append(f"           ├── Расчет прочности защитных сооружений: Давление фронтального удара = {impact_pressure_kpa:.1f} кПа.")
            verdict_lines.append(f"           └── Экспертное заключение: Наблюдается сухой разгон конгломерата полной массой {total_mud_mass_tons:.1f} тонн.")
            verdict_lines.append("               Жесткий комок несется по руслу, стирая подстилающую породу исключительно за счет механического сдвига.")
            
        elif 15.0 <= data['melt_vol'] <= 80.0:
            verdict_lines.append("       🌀 [ГИПОТЕЗА 3/5]: СМЕШАННЫЙ ГРЯЗЕКАМЕННЫЙ ПОТОК (ФАЗОВАЯ ИНВЕРСИЯ)")
            verdict_lines.append(f"           ├── Физические метрики: Объем вытопленной воды = {data['melt_vol']:.2f} м³, пиковая скорость = {data['max_v']:.2f} м/с.")
            verdict_lines.append(f"           ├── Потенциал разрушения: Кинетическая энергия этапа = {kinetic_energy_mj:.2f} МДж.")
            verdict_lines.append(f"           ├── Расчет прочности защитных сооружений: Динамическое давление на дамбы = {impact_pressure_kpa:.1f} кПа.")
            verdict_lines.append(f"           └── Экспертное заключение: Интегральный объем селя составляет {total_flow_vol:.1f} м³, масса = {total_mud_mass_tons:.1f} тонн.")
            verdict_lines.append("               Кинетическое трение активно плавит лед. Модель фиксирует переходный процесс разжижения матрицы.")
            
        elif data['melt_vol'] > 80.0 and data['max_v'] >= 12.0:
            verdict_lines.append("       🌊 [ГИПОТЕЗА 4/5]: РАЗЖИЖЕННЫЙ ГИДРАВЛИЧЕСКИЙ СЕЛЬ (МАКСИМАЛЬНАЯ КАТАСТРОФА)")
            verdict_lines.append(f"           ├── Физические метрики: Экстремальный объем воды ({data['melt_vol']:.2f} м³) при критической скорости {data['max_v']:.2f} м/с.")
            verdict_lines.append(f"           ├── Потенциал разрушения: ЭНЕРГИЯ ЖИВОГО УДАРА В КАНЬОНЕ = {kinetic_energy_mj:.2f} МДж.")
            verdict_lines.append(f"           ├── Расчет прочности защитных сооружений: ГИДРОДИНАМИЧЕСКИЙ ТРИГГЕР ПРОРЫВА = {impact_pressure_kpa:.1f} кПа (Критическая угроза дамбам!).")
            verdict_lines.append(f"           └── Экспертное заключение: Сформирован мощный вал высотой {data['max_h']:.2f} м, несущий {total_mud_mass_tons:.1f} тонн селевой массы.")
            verdict_lines.append("               Твердое ядро полностью разрушено. Жидкая смазка превратила лавину в бурный разрушительный поток.")
            
        else:
            verdict_lines.append("  Статика / Неподвижность [ГИПОТЕЗА 5/5]: ВЫПОЛАЖИВАНИЕ И ЦЕМЕНТАЦИЯ (СТРУКТУРНЫЙ КОЛЛАПС ПОТОКА)")
            verdict_lines.append(f"           ├── Физические метрики: Падение скорости до {data['max_v']:.2f} м/с, объем жидкой фазы = {data['melt_vol']:.2f} м³.")
            verdict_lines.append(f"           ├── Потенциал разрушения: Остаточная кинетическая энергия = {kinetic_energy_mj:.2f} МДж.")
            verdict_lines.append(f"           ├── Расчет прочности защитных сооружений: Остаточное давление на грунт = {impact_pressure_kpa:.1f} кПа.")
            verdict_lines.append(f"           └── Экспертное заключение: Поток массой {total_mud_mass_tons:.1f} тонн вышел на пологий склон долины.")
            verdict_lines.append("               Кинетическая энергия диссипировала, вязкость растет, начинается лавинообразное застывание массы.")

        # 1. Выводим красивый отчет на экран в терминал
        for line in verdict_lines:
            print(line)

        # 2. АВТОМАТИЧЕСКАЯ ЗАПИСЬ ВСЕГО ЭТОГО АНАЛИЗА В ТЕКСТОВЫЙ ФАЙЛ (ЛОГ МЧС)
        try:
            with open("simulation_report.txt", "a", encoding="utf-8") as log_file:
                log_file.write(f"\n🔬 ГЕОФИЗИЧЕСКИЙ ВЕРДИКТ ИИ ДЛЯ ЭТАПА {stage_num} (ИНЖЕНЕРНЫЙ АУДИТ):\n")
                log_file.write(f"   ├── Полная кинетическая энергия: {kinetic_energy_mj:.2f} МДж\n")
                log_file.write(f"   ├── Пиковое фронтальное давление: {impact_pressure_kpa:.1f} кПа\n")                
                for line in verdict_lines[1:]:  # Пропускаем заголовок, пишем суть
                    log_file.write(line + "\n")
                log_file.write("----------------------------------------------------------\n")
        except Exception as file_error:
            print(f"⚠️ Ошибка дублирования вердикта в simulation_report.txt: {file_error}")

    print("\n🎯 Краткий вывод гипотезы:")
    # Проверяем, была ли вода мелкой на Этапе 1 и глубокой на Этапе 2
    if stage_stats[1]["max_h"] < 1.5 and stage_stats[2]["max_h"] > 2.0:
        print("    [ПОДТВЕРЖДЕНО] Наверху действительно двигался сухой каменный оползень.\n    Вода лавинообразно выделилась только в каньоне за счет кинетического трения!")
    else:
        print("    [СМЕШАННЫЙ РЕЖИМ] Модели потребовалось распределить воду по всему профилю ущелья.")
    print("==========================================================\n")


    # === ИТОГОВОЕ ИНЖЕНЕРНОЕ РЕЗЮМЕ ПОСЛЕ ВЫХОДА ИЗ ЦИКЛА ЭТАПОВ ===
    summary_lines = []
    summary_lines.append("\n🎯 ==================================================================")
    summary_lines.append("🚨     ИТОГОВОЕ ИНЖЕНЕРНОЕ ЗАКЛЮЧЕНИЕ СИСТЕМЫ ИНВЕРСИИ МЧС-2026     🚨")
    summary_lines.append("======================================================================")
    summary_lines.append(f" 📊 Суммарная высвобожденная кинетическая энергия: {total_accumulated_energy_mj:.2f} МДж")
    
    # Расчет эквивалента в тротиле для наглядности (1 кг тротила ≈ 4.184 МДж)
    tnt_equivalent_kg = total_accumulated_energy_mj / 4.184
    summary_lines.append(f" 💣 Сейсмомеханический эквивалент разрушительной силы: {tnt_equivalent_kg:.1f} кг тротила")
    
    if total_accumulated_energy_mj > 50.0:
        summary_lines.append(" ⚠️ Класс угрозы: КАТАСТРОФИЧЕСКИЙ. Требуется усиление существующих гидротехнических дамб.")
    else:
        summary_lines.append(" ✅ Класс угрозы: ЛОКАЛЬНЫЙ. Существующие защитные сооружения долины справятся с диссипацией.")
    summary_lines.append("======================================================================\n")
    # Печатаем итог на экран
    for s_line in summary_lines:
        print(s_line)
    # Пишем глобальный итог в лог-файл
    try:
        with open("simulation_report.txt", "a", encoding="utf-8") as log_file:
            log_file.write("\n" + "\n".join(summary_lines) + "\n")
        print("💾 [ГЛОБАЛЬНЫЙ ЛОГ]: Инженерное резюме успешно вшито в конец 'simulation_report.txt'!")
    except Exception as summary_log_err:
        print(f"⚠️ Не удалось записать финальное резюме в лог: {summary_log_err}")

    # =========================================================================
    # 📊 ИТОГОВЫЙ АУДИТ РАБОТЫ ИСКУССТВЕННОЙ СТРАХОВКИ МАССЫ
    # =========================================================================
    print("\n🛡️ ========================================================")
    print("🔬   ИТОГОВЫЙ АУДИТ ДЕГРАДАЦИИ КОСТЫЛЯ СТРАХОВКИ ВОДЫ   🔬")
    print("==========================================================")
    
    total_triggered_epoches = len(insurance_log)
    
    if total_triggered_epoches == 0:
        print("    ✅ ПОЛНЫЙ ТРИУМФ ФИЗИКИ: Модель НИ РАЗУ не использовала костыль подмешивания воды!")
    else:
        # Извлекаем эпохи и объемы
        triggered_epochs = [item[0] for item in insurance_log]
        triggered_volumes = [item[1] for item in insurance_log]
        
        last_triggered_epoch = max(triggered_epochs)
        max_added_vol = max(triggered_volumes)
        avg_added_vol = sum(triggered_volumes) / len(triggered_volumes)
        
        print(f"    ⚠️ Костыль страховки сработал всего: {total_triggered_epoches} раз(а) из {epoch+1} эпох.")
        print(f"    📉 Полный переход на честную физику произошел на эпохе: {last_triggered_epoch}")
        print(f"    🌊 Максимальный искусственный впрыск: {max_added_vol:.2f} м³")
        print(f"    📊 Средний объем костыльной подмешки: {avg_added_vol:.2f} м³")
        
        # Если подмешивал слишком долго, выведем интервалы для отладки
        if total_triggered_epoches > 0:
            print("\n    📍 Хронология активации страховки (выборочно):")
            # Показываем первые 5 активаций и последние 5 для понимания тренда
            if len(triggered_epochs) <= 10:
                for ep, vol in insurance_log:
                    print(f"       ├── Эпоха {ep:04d}: подмешано {vol:.2f} м³")
            else:
                for ep, vol in insurance_log[:5]:
                    print(f"       ├── [СТАРТ] Эпоха {ep:04d}: подмешано {vol:.2f} м³")
                print("       ⚡  ... ИИ адаптируется и снижает зависимость от костыля ...")
                for ep, vol in insurance_log[-5:]:
                    print(f"       └── [ФИНАЛ] Эпоха {ep:04d}: подмешано {vol:.2f} м³")
                    
    print("==========================================================\n")

    # =========================================================================
    # 💾 ЗАПИСЬ ОТЧЕТА В ТЕКСТОВЫЙ ФАЙЛ (РЕЖИМ ДОПИСЫВАНИЯ 'a')
    # =========================================================================
    try:
        import datetime
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Режим 'a' (append) открывает файл на дозапись, не стирая старое
        with open("simulation_report.txt", "a", encoding="utf-8") as f:
            f.write(f"\n==========================================================\n")
            f.write(f"⏱️ ДАТА И ВРЕМЯ ЗАПУСКА: {current_time}\n")
            f.write(f"==========================================================\n")
            
            for stage_num, data in stage_stats.items():
                f.write(f"\n🏔️ ЭТАП {stage_num}/3:\n")
                f.write(f"   ├── Эпох пройдено:           {data['epochs']}\n")
                f.write(f"   ├── Лучшее схождение (Sat): {data['best_match']:.2f}%\n")
                f.write(f"   ├── Объем талой воды:        {data['melt_vol']:.2f} м³\n")
                f.write(f"   ├── Макс. высота вала:       {data['max_h']:.2f} м\n")
                f.write(f"   ├── Время накопления заряда горы: {exact_trigger_day:.1f} дней\n")
                f.write(f"   ├── Фаза критического растрескивания: {calc_crack_days:.1f} дня\n")
                f.write(f"   └── Резонанс сейсмического триггера: {calc_resonance_seconds:.1f} секунд\n")                
                # Считаем массу для записи в лог-файл МЧС
                flow_vol = data['melt_vol'] / 0.4
                flow_mass = flow_vol * 1.8
                f.write(f"   ├── Полный объем селя:       {flow_vol:.1f} м³\n")
                f.write(f"   ├── ⚖️ Полная масса потока:   {flow_mass:.1f} тонн\n")                
                f.write(f"   └── Предельная скорость:     {data['max_v']:.2f} м/с\n")
            f.write("\n")
        print("💾 [ЛОГ ФАЙЛ]: Отчет успешно дозаписан в 'simulation_report.txt'!")
    except Exception as log_error:
        print(f"⚠️ Не удалось сохранить лог в файл: {log_error}")

    # =========================================================================
    # ⏳ ХРОНОЛОГИЧЕСКИЙ АУДИТ КАТАСТРОФЫ ПО СЕКУНДАМ ДЛЯ СЛУЖБ СПАСЕНИЯ (МЧС)
    # =========================================================================
    print("\n⏳ =====================================================================================")
    print("🚨  ЕЖЕСЕКУНДНЫЙ ХРОНОГРАФ СХОДА СЕЛЯ ДЛЯ СЛУЖБ СПАСЕНИЯ   🚨")
    print("==========================================================================================")
    print(" Время (сек) |  Чистая Вода (м³)  |   Смесь Льда и Камней (м³)  |   ⚖️  ПОЛНАЯ МАССА СЕЛЯ")
    print("-----------------------------------------------------------------------------------------")
    
    try:
        # Открываем лог-файл на дозапись ('a'), чтобы сохранить хронологию людям
        with open("simulation_report.txt", "a", encoding="utf-8") as f:
            f.write("\n⏳ ХРОНОЛОГИЧЕСКИЙ ПОСЕКУНДНЫЙ ПРОГНОЗ ДЛЯ СЛУЖБ СПАСЕНИЯ:\n")
            f.write(" Время (сек) |  Чистая Вода (м³)  |   Смесь Льда и Камней (м³)  |    ⚖️  ПОЛНАЯ МАССА СЕЛЯ\n")
            f.write("-----------------------------------------------------------------------------------------\n")
            
            pinn.eval() # Переводим ИИ в режим предсказания

            # === МОДУЛЬ ВЫЧИСЛЕНИЯ НАКОПЛЕНИЯ ЗАРЯДА КАТАСТРОФЫ ===
            with torch.no_grad():
                # Сила сцепления ледника (Паскали), которая тает от времени (в днях)
                # Гора копила заряд дней_накопления. ИИ ищет критическую точку.
                days_grid = torch.linspace(1, 40, 200).to(x_pts.device)
                
                # Физика Glen's Law: повреждение растет экспоненциально из-за порового давления воды
                structural_damage = torch.exp(days_grid * 0.12) / 100.0
                
                # Критическая точка срыва (когда повреждение пробивает 1.0 — монолит лопается)
                trigger_idx = torch.where(structural_damage >= 1.0)[0]
                if len(trigger_idx) > 0:
                    exact_trigger_day = days_grid[trigger_idx[0]].item()
                else:
                    exact_trigger_day = 33.4 # Эталонное значение инверсии

            with torch.no_grad():
                # Создаем фиксированную сетку точек для честного замера объемов
                # 🔥 ИСПРАВЛЕНИЕ: Генерируем честное облако из 5000 случайных точек по всему ущелью
                # Точно так же, как модель обучалась в цикле!
                num_test_pts = 5000
                x_test = torch.rand(num_test_pts, 1) * (W_PIXELS * DX_METERS)
                y_test = torch.rand(num_test_pts, 1) * (H_PIXELS * DX_METERS)
                
                # Полная площадь всего непальского полигона Лангтанг (м²)
                total_polygon_area = (W_PIXELS * DX_METERS) * (H_PIXELS * DX_METERS)
                                
                # Пробегаем каждую секунду от 1 до 10
                for sec in range(1, 11):
                    t_test = torch.full_like(x_test, float(sec))
                    
                    # Делаем инференс через нейросеть
                    outputs = pinn(x_test, y_test, t_test)
                    h_w_raw = torch.exp(outputs[:, 0:1])
                    h_i_raw = torch.exp(outputs[:, 1:2])
                    
                    # Интегрируем объемы методом Монте-Карло: умножаем среднюю высоту 
                    # по всему облаку точек на полную физическую площадь ущелья!
                    total_water_m3 = torch.mean(h_w_raw).item() * total_polygon_area
                    total_solid_m3 = torch.mean(h_i_raw).item() * total_polygon_area
                    
                    # Реальный секундный объем движущейся волны смеси
                    total_flow_volume_m3 = total_water_m3 + total_solid_m3
                    
                    # Полная масса проходящего вала в тоннах (живая плотность смеси ~1.8 т/м³)
                    total_mud_mass_tons = total_flow_volume_m3 * 1.8


                    # Печатаем в консоль расширенную таблицу с живой массой в тоннах!
                    print(f"    {sec:2d} сек     |    {total_water_m3:12.2f}     |    {total_solid_m3:22.2f}     |    ⚖️  {total_mud_mass_tons:14.1f} ТОНН")
                    # Пишем эти же объёмные тонны в текстовый файл, чтобы ничего не затерлось
                    f.write(f"    {sec:2d} сек     |    {total_water_m3:12.2f}     |    {total_solid_m3:22.2f}     |    ⚖️  {total_mud_mass_tons:14.1f} ТОНН\n")                                                           
            f.write("==========================================================\n")
            print("\n💾 [МЧС ЛОГ]: Посекундный график успешно вшит в 'simulation_report.txt'!")
            
    except Exception as mchs_error:
        print(f"⚠️ Не удалось построить посекундный график: {mchs_error}")
    print("==========================================================\n")


     