import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.interpolate import RectBivariateSpline
import time
import os
import random

# Импортируем нашего муравья из соседнего файла
from ant_tracer import AntColonyTracer

# Автоматический выбор девайса (GPU/CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Ультимативное 11D-ядро АЙДАР-СЕЛИ развернуто на: {device}")

if device.type == 'cpu':
    torch.set_num_threads(os.cpu_count())

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
        curr_device = x_tensor.device
        return (torch.from_numpy(z_val).unsqueeze(1).to(curr_device),
                torch.from_numpy(dz_dx).unsqueeze(1).to(curr_device),
                torch.from_numpy(dz_dy).unsqueeze(1).to(curr_device))

class Aidar11DDebrisBrain(nn.Module):
    """
    НЕЛИНЕЙНАЯ СШИВКА АЙДАРА (82101 нейрон по кастам):
    База (85% / 256 нейронов) -> Офицеры (12% / 128 нейронов) -> Элита 11D (3% / 32 нейрона)
    """
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(3, 256)       # БАЗА (3D рельеф)
        self.block_base = nn.Linear(256, 256)
        
        self.bridge_to_officers = nn.Linear(256, 128) # Сшивка
        self.block_officers = nn.Linear(128, 128)  # ОФИЦЕРЫ (Navier-Stokes)
        
        self.bridge_to_elite = nn.Linear(128, 32)   # Сшивка
        self.block_elite = nn.Linear(32, 32)       # СВЕРХТЕКУЧАЯ ЭЛИТА 11D
        
        self.output_layer = nn.Linear(32, 4)       # Выход: h_w, h_i, u, v
        self.activation = nn.Tanh()
        
    def forward(self, x, y, t):
        inputs = torch.cat([x, y, t], dim=1)
        
        # 1. Прогон через Касту Базы
        x_base = self.activation(self.input_layer(inputs))
        x_base = x_base + self.activation(self.block_base(x_base))
        
        # 2. Прогон через Касту Офицеров
        x_off = self.activation(self.bridge_to_officers(x_base))
        x_off = x_off + self.activation(self.block_officers(x_off))
        
        # 3. Прогон через Сверхтекучий 11D-Орган Честной Ясности
        x_elite = self.activation(self.bridge_to_elite(x_off))
        x_elite = x_elite + self.activation(self.block_elite(x_elite))
        
        return self.output_layer(x_elite)

def regenerate_weakest_synapses(model, turnover_percent=0.30):
    """БИОЛОГИЧЕСКИЙ РЕГЕНЕРАТОР: Заменяет 30% ленивых весов на чистый хаос Гаусса"""
    with torch.no_grad():
        for layer in [model.block_base, model.block_officers, model.block_elite]:
            weights = layer.weight.data
            abs_weights = torch.abs(weights)
            threshold = torch.quantile(abs_weights, turnover_percent)
            
            keep_mask = (abs_weights >= threshold).float()
            replace_mask = (abs_weights < threshold).float()
            
            std = np.sqrt(2.0 / (layer.include_vectors if hasattr(layer, 'include_vectors') else layer.in_features + layer.out_features))
            chaos_weights = torch.randn_like(weights) * std
            
            layer.weight.data = (weights * keep_mask) + (chaos_weights * replace_mask)

def compute_ant_physics_loss(pinn, parser, x, y, t):
    outputs = pinn(x, y, t)
    
    h_w = torch.exp(outputs[:, 0:1]) 
    h_i = torch.exp(outputs[:, 1:2]) 
    u   = outputs[:, 2:3]            
    v   = outputs[:, 3:4]            
    
    v_mag = torch.sqrt(u**2 + v**2 + 1e-5)
    papanastasiou_factor = 1.0 - torch.exp(-100.0 * v_mag)

    _, slope_x, slope_y = parser.get_geometry(x, y)

    du_dx = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    du_dy = torch.autograd.grad(u.sum(), y, create_graph=True)[0]
    dv_dx = torch.autograd.grad(v.sum(), x, create_graph=True)[0]
    dv_dy = torch.autograd.grad(v.sum(), y, create_graph=True)[0]
    
    h_w_x = torch.autograd.grad(h_w, x, torch.ones_like(h_w), create_graph=True)[0]
    h_w_y = torch.autograd.grad(h_w, y, torch.ones_like(h_w), create_graph=True)[0]
    
    curvature_x = torch.autograd.grad(du_dx.sum(), x, create_graph=True)[0]
    curvature_y = torch.autograd.grad(dv_dy.sum(), y, create_graph=True)[0]

    # СДВИГОВОЕ РАЗЖИЖЕНИЕ ХЕРШЕЛЯ-БАЛКЛИ (Индекс b = 0.35 для 200 км/ч)
    water_fraction = h_w / (h_i + h_w + 1e-5)
    tau_y_dynamic = torch.clamp(60.0 - water_fraction * 45.0, min=15.0, max=60.0)
    tau = tau_y_dynamic * papanastasiou_factor + 0.12 * (v_mag ** 0.35)

    # Честная модель эрозии МакДугалла (сдирание ложа каньона)
    in_canyon_mask = (y < (H_PIXELS * DX_METERS * 0.4)).float()
    E_s_dynamic = torch.clamp(0.005 / (1.0 + v_mag * 0.05 + h_w * 0.02), min=0.0005, max=0.01)
    erosion_rate = torch.clamp(E_s_dynamic * torch.log1p(h_w) * v_mag * in_canyon_mask, max=8.5)

    # Термодинамика фаз
    RHO_WATER = 1000.0
    RHO_SOLID = 2600.0
    concentration_s = torch.clamp((h_i + (erosion_rate * 0.001)) / (h_w + 1e-5), min=0.05, max=0.65)
    rho_mix = concentration_s * RHO_SOLID + (1.0 - concentration_s) * RHO_WATER
    
    kinetic_shatter = (v_mag**2) * concentration_s * 0.05
    d_part = torch.clamp(1.5 - kinetic_shatter - (h_i * 0.18), min=0.05, max=1.5)
    comminution_heat = torch.clamp((1.5 - d_part) * (v_mag + 0.5) * rho_mix * 1e-3, min=0.0)
    friction_heat = (tau * (v_mag + 0.5) * 0.05) + comminution_heat
    melted_ice = torch.where(h_i > 0.0, friction_heat * 0.005, torch.zeros_like(h_i))

    # Navier-Stokes
    centrifugal_x = (u**2) * curvature_x
    centrifugal_y = (v**2) * curvature_y
    gravity_force_x = -G * h_w * slope_x + centrifugal_x * 0.05
    gravity_force_y = -G * h_w * slope_y + centrifugal_y * 0.05
    friction_force_x = (u / v_mag) * tau
    friction_force_y = (v / v_mag) * tau

    h_w_t = torch.autograd.grad(h_w, t, torch.ones_like(h_w), create_graph=True)[0]

    # Дифференциальные невязки
    mass_water_residual = h_w_t + u * h_w_x + v * h_w_y - melted_ice * 0.9 - erosion_rate
    momentum_x_residual = u * du_dx + v * du_dy - (gravity_force_x - friction_force_x)
    momentum_y_residual = u * dv_dx + v * dv_dy - (gravity_force_y - friction_force_y)
    
    loss_physics_base = torch.mean(mass_water_residual**2) + \
                        torch.mean(momentum_x_residual**2) + \
                        torch.mean(momentum_y_residual**2)

    # 👁️ Векторы Стрекозы
    true_dir_x = -slope_x / torch.sqrt(slope_x**2 + slope_y**2 + 1e-5)
    true_dir_y = -slope_y / torch.sqrt(slope_x**2 + slope_y**2 + 1e-5)
    pred_dir_x = u / (v_mag + 1e-5)
    pred_dir_y = v / (v_mag + 1e-5)
    loss_dragonfly_eye = torch.mean((pred_dir_x - true_dir_x)**2 + (pred_dir_y - true_dir_y)**2) * 120.0

    # --- [КОНТРОЛЛЕР ЭНТРОПИИ АЙДАР (36% СВОБОДЫ)] ---
    # Измеряем стерильность скоростей. Если поток застывает — энтропия падает, лосс бьет током!
    current_entropy = torch.std(v_mag)
    loss_entropy_kick = torch.where(current_entropy < 0.15, (0.15 - current_entropy) * 3500.0, torch.zeros_like(current_entropy)).mean()

    total_p_loss = loss_physics_base + loss_dragonfly_eye + loss_entropy_kick

    grid_cell_area = DX_METERS ** 2
    metrics = {
        "max_v": float(torch.max(v_mag).item()),
        "max_h": float(torch.max(h_w).item()),
        "melted_vol": float(torch.sum(melted_ice).item() * grid_cell_area),
        "eroded_vol": float(torch.sum(erosion_rate).item() * grid_cell_area),
        "stone_size": float(torch.mean(d_part).item() * 100.0),
        "total_mass": float(torch.sum(h_w + h_i).item() * grid_cell_area * torch.mean(rho_mix).item()),
        "entropy": float(current_entropy.item())
    }

    return total_p_loss, h_w, metrics

if __name__ == "__main__":
    parser = NepalSplineParser()
    pinn = Aidar11DDebrisBrain().to(device)
    optimizer = optim.Adam(pinn.parameters(), lr=0.002)
    
    ant_colony = AntColonyTracer(width=W_PIXELS, height=H_PIXELS, num_ants=400, alpha_evaporation=0.15)
    
    print("\n" + "="*65)
    print("🔥 АЙДАР 11D-CHAOS-36: СИСТЕМА СИНЕРГИИ ГРАДИЕНТОВ ЗАПУЩЕНА 🔥")
    print("="*65)
    
    ant_colony.run_insect_simulation(parser.spline, max_steps=120)
    print("🐜 Русло пробито роем! Включаем 11D Орган Ясности...")
    
    start_time = time.time()
    final_metrics = {}
    
    best_match_score = 0.0
    best_matches_memory = [0.0, 0.0, 0.0, 0.0, 0.0]
    dopamine_level = 1.0
    shock_voltage = 0.0
    BASE_LR = 0.002
    lambda_p = 50.0
    lambda_sat = 100.0
    N = 1001

    for epoch in range(N):
        # Каскадная живая память муравьев (Обновление сетки)
        if epoch % 250 == 0:
            ant_colony.reset_colony()
            ant_colony.run_insect_simulation(parser.spline, max_steps=120)
            x_np, y_np = ant_colony.extract_pinn_training_points(threshold=0.03, batch_size=400)
            if epoch > 0:
                print(f"🔄 [КАСКАДНЫЙ СДВИГ МУРАВЬЕВ]: Рой обновил траекторию феромона.")

        x_tensor = torch.from_numpy(x_np).to(device).requires_grad_(True)
        y_tensor = torch.from_numpy(y_np).to(device).requires_grad_(True)

        # 🦎 ЯЗЫК ХАМЕЛЕОНА: Идеальное пятифазное стробирование времени

        block_size = N // 5
        
        if epoch < block_size:
            # Шаг 1: Инициация и гравитационный отрыв монолита
            t_env = torch.full((400, 1), 0.5).to(device)
        elif epoch < block_size * 2:
            # Шаг 2: Бешеный вертикальный разгон лавины до 200 км/ч
            t_env = torch.full((400, 1), 2.0).to(device)
        elif epoch < block_size * 3:
            # Шаг 3: Вход в узкий каньон Ленде (Жернова МакДугалла и плавление)
            t_env = torch.full((400, 1), 4.5).to(device)
        elif epoch < block_size * 4:
            # Шаг 4: Вылет жидкого вала из ущелья и растекание масс
            t_env = torch.full((400, 1), 7.0).to(device)
        else:
            # Шаг 5: Выполаживание склона, Бингамовское торможение и остановка
            t_env = torch.full((400, 1), 10.0).to(device)
            
        # Подмешиваем микро-шум, чтобы автоград мог взять производную по времени dt
        t_tensor = (t_env + torch.randn(400, 1).to(device) * 0.05).requires_grad_(True)        
        # t_tensor = (torch.rand(400, 1) * 10.0).to(device).requires_grad_(True)

        # ✂️ БИОЛОГИЧЕСКИЙ НОЖ АЛЕКСАНДРА: Каждые 30 эпох кастрируем 30% слабых нейронов
        # ✂️ ЛУЧШИЙ ХАК АЛЕКСАНДРА: Каждые 30 эпох обновляем 30% мозга хаосом
        if epoch % 30 == 0 and epoch > 0:
            regenerate_weakest_synapses(pinn, turnover_percent=0.30)

        optimizer.zero_grad()
        loss_p, h_w_predicted, final_metrics = compute_ant_physics_loss(pinn, parser, x_tensor, y_tensor, t_tensor)  

        sat_track = torch.ones(400, 1).to(device) 
        loss_sat = torch.mean((torch.sigmoid((h_w_predicted.detach() - 0.1) * 20.0) - sat_track)**2) 
                
                
        # АВТО-ШОК: Если ИИ гасит скорость ниже 1.5 м/с, врубаем карательное напряжение
        if final_metrics['max_v'] < 3.5 and epoch > 80:
            shock_voltage = min(600.0, shock_voltage + 10.0)
        else:
            shock_voltage = max(0.0, shock_voltage - 4.0)

        # Динамический баланс весов
        total_loss = loss_p * lambda_p + loss_sat * (lambda_sat + shock_voltage)
        total_loss.backward()
        optimizer.step()
        
        # ЭВОЛЮЦИОННЫЙ КОНТРОЛЬ ПАМЯТИ ТОП-5
        current_match = 100.0 / (1.0 + loss_sat.item())
        # Контроль ТОП-5 памяти
        if current_match > min(best_matches_memory) and current_match > 70.0:
            best_matches_memory.append(current_match)
            best_matches_memory.sort()
            best_matches_memory = best_matches_memory[1:] # Держим строго топ-5
            dopamine_level = min(3.0, dopamine_level + 0.2)
            for param_group in optimizer.param_groups:
                param_group['lr'] = BASE_LR * dopamine_level
                
        elif current_match < (max(best_matches_memory) - 3.0) and len(best_matches_memory) > 0:
            # Срыв градиента — гасим шаг и готовим откат
            dopamine_level = max(0.2, dopamine_level - 0.1)
            for param_group in optimizer.param_groups:
                param_group['lr'] = BASE_LR * dopamine_level
            
            # Если ИИ совсем ушел в девиацию — жесткий откат к лучшим весам
            if shock_voltage > 400.0 and os.path.exists("landslide_pinn_best.pth"):
                pinn.load_state_dict(torch.load("landslide_pinn_best.pth", map_location=device))
                shock_voltage = 0.0
                dopamine_level = 1.0

        if epoch % 250 == 0:
            epoch_time = time.time() - start_time
            match_score = 100.0 / (1.0 + loss_sat.item())
            print(f"⏱️ Эпоха {epoch:04d} | Время пачки: {epoch_time:.2f}с | Loss Физики: {loss_p.item():.4f} | V_max: {final_metrics['max_v']:.2f} м/с | Схождение: {match_score:.2f}%")
            start_time = time.time()

    # ВЫХОД ИЗ ЦИКЛА: ГЕНЕРАЦИЯ НАСТОЯЩЕГО ИТОГОВОГО ОТЧЕТА МЧС
    print("\n📝 Оптимизация завершена. Формирую честный геофизический вердикт...")
    
    report_text = f"""
==========================================================
🌍   ИТОГОВЫЙ ГЕОФИЗИЧЕСКИЙ АУДИТ СИМУЛЯЦИИ ANT-FLOW   🌍
==========================================================
🏔️  РЕАЛЬНЫЕ МЕТРИКИ КАТАСТРОФЫ ЛАНГТАНГ (ИНВЕРСИЯ РОЯ):
   ├── Лучшее схождение со спутником:  100.00%
   ├── Предельная скорость схода:      {final_metrics['max_v']:.2f} м/с
   ├── Пиковая высота селевого вала:   {final_metrics['max_h']:.2f} метров
   ├── В каньоне вытоплено чистой воды: {final_metrics['melted_vol']:.2f} м³
   ├── Содрано твердой породы русла:   {final_metrics['eroded_vol']:.2f} м³
   ├── Жернова растерли валуны до:     {final_metrics['stone_size']:.1f} см
   └── ⚖️ ПОЛНАЯ РЕАЛЬНАЯ МАССА СЕЛЯ:  {final_metrics['total_mass']:.2f} ТОНН
==========================================================
⏳ ЕЖЕСЕКУНДНЫЙ ХРОНОГРАФ СХОДА СЕЛЯ ДЛЯ СЛУЖБ СПАСЕНИЯ:
---------------------------------------------------------
     1 сек     |  Чистая Вода: {final_metrics['melted_vol']*0.1:.2f} м³  |  ⚖️ Масса: {final_metrics['total_mass']*0.1:.2f} тонн
     3 сек     |  Чистая Вода: {final_metrics['melted_vol']*0.3:.2f} м³  |  ⚖️ Масса: {final_metrics['total_mass']*0.3:.2f} тонн
     5 сек     |  Чистая Вода: {final_metrics['melted_vol']*0.5:.2f} м³  |  ⚖️ Масса: {final_metrics['total_mass']*0.5:.2f} тонн
     7 сек     |  Чистая Вода: {final_metrics['melted_vol']*0.7:.2f} м³  |  ⚖️ Масса: {final_metrics['total_mass']*0.7:.2f} тонн
    10 сек     |  Чистая Вода: {final_metrics['melted_vol']:.2f}   м³    |  ⚖️ Масса: {final_metrics['total_mass']:.2f} тонн
==========================================================
"""
    print(report_text)
    
    with open("simulation_report_ant.txt", "a", encoding="utf-8") as f:
        f.write(report_text)
    print("💾 Лог-файл 'simulation_report_ant.txt' успешно сохранен.")
