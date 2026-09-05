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

class LandslidePINNCore(nn.Module):
    def __init__(self):
        super().__init__()
        # Увеличиваем количество нейронов до 128 и добавляем 4-й слой для нелинейной емкости
        self.net = nn.Sequential(
            nn.Linear(3, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 4)
        )
        
    def forward(self, x, y, t):
        inputs = torch.cat([x, y, t], dim=1)
        return self.net(inputs)

def compute_nepal_physics_loss(pinn, parser, x, y, t, tau_y, K, melt_rate):
    outputs = pinn(x, y, t)
    
    # Нормализация размерностей (Разбиваем на метры и секунды)
    h_w_raw = torch.exp(outputs[:, 0:1])
    h_i     = torch.exp(outputs[:, 1:2])
    u       = outputs[:, 2:3]
    v       = outputs[:, 3:4]
    
    v_mag = torch.sqrt(u**2 + v**2 + 1e-5)
    tau = tau_y + K * v_mag
    friction_heat = tau * v_mag * 0.005
    
    expected_melted_vol = torch.sum(torch.where(h_i > 0.0, friction_heat * melt_rate, torch.zeros_like(h_i)))
    total_raw_water_vol = torch.sum(h_w_raw) + 1e-8
    
    h_w = h_w_raw * (expected_melted_vol / total_raw_water_vol)
    _, slope_x, slope_y = parser.get_geometry(x, y)

    # ⏱️ Элемент дробного дифференциала (Масштабирование памяти времени)
    # Задаем альфа-коэффициент затухания для разных временных масштабов
    alpha_memory = 0.8 
    t_scaled = t ** alpha_memory

    h_w_t = torch.autograd.grad(h_w, t, torch.ones_like(h_w), create_graph=True)[0] * t_scaled
    h_i_t = torch.autograd.grad(h_i, t, torch.ones_like(h_i), create_graph=True)[0] * t_scaled
    u_t   = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0] * t_scaled
    v_t   = torch.autograd.grad(v, t, torch.ones_like(v), create_graph=True)[0] * t_scaled

    h_w_x = torch.autograd.grad(h_w, x, torch.ones_like(h_w), create_graph=True)[0]
    h_w_y = torch.autograd.grad(h_w, y, torch.ones_like(h_w), create_graph=True)[0]

    friction_force_x = (u / v_mag) * tau
    friction_force_y = (v / v_mag) * tau
    gravity_force_x = -G * h_w * slope_x
    gravity_force_y = -G * h_w * slope_y
    
    melted_ice = torch.where(h_i > 0.0, friction_heat * melt_rate, torch.zeros_like(h_i))

    # Масштабируем невязки (уменьшаем абсолютные значения Паскалей до порядка единиц)
    mass_water_residual = (h_w_t + u * h_w_x + v * h_w_y - melted_ice * 0.9) * 0.01
    mass_ice_residual = (h_i_t + melted_ice) * 0.01
    momentum_x_residual = (u_t + u * h_w_x - (gravity_force_x - friction_force_x)) * 0.001
    momentum_y_residual = (v_t + v * h_w_y - (gravity_force_y - friction_force_y)) * 0.001

    loss_p = torch.mean(mass_water_residual**2) + \
             torch.mean(mass_ice_residual**2) + \
             torch.mean(momentum_x_residual**2) + \
             torch.mean(momentum_y_residual**2)

    # === ВСТАВИТЬ ПЕРЕД return loss_p, h_w ===
    if torch.rand(1).item() < 0.01:  # Печатаем случайно в 1% случаев, чтобы не спамить в консоль
        print("\n--- [Диагностика физики Сен-Венана] ---")
        print(f"  💧 Ошибка массы воды:  {torch.mean(mass_water_residual**2).item():.6f}")
        print(f"  ❄️ Ошибка массы льда:  {torch.mean(mass_ice_residual**2).item():.6f}")
        print(f"  🏎️ Ошибка импульса X:  {torch.mean(momentum_x_residual**2).item():.6f}")
        print(f"  🏎️ Ошибка импульса Y:  {torch.mean(momentum_y_residual**2).item():.6f}")
        print(f"  🌊 Макс. высота воды:  {torch.max(h_w).item():.2f} м | Макс. скорость: {torch.max(v_mag).item():.2f} м/с")
    
    # 🛠️ ЗАПЛАТКА: Штрафуем модель, если глубина движущегося потока ниже 1.0 метра
    # Если скорость > 1.0 м/с, а высота < 1.0 метра — включается квадратичный штраф
    low_water_penalty = torch.where(
        (v_mag > 1.0) & (h_w < 1.0), 
        (1.0 - h_w) ** 2, 
        torch.zeros_like(h_w)
    )
    loss_low_water = torch.mean(low_water_penalty) * 100.0 # Вес заплатки

    # Добавляем этот штраф к общему физическому лоссу
    loss_p = torch.mean(mass_water_residual**2) + \
             torch.mean(mass_ice_residual**2) + \
             torch.mean(momentum_x_residual**2) + \
             torch.mean(momentum_y_residual**2) + \
             loss_low_water

    return loss_p, h_w

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
    lambda_p = 1.0
    lambda_sat = 50.0 # Поднимаем базовый вес спутника, чтобы компенсировать разницу масштабов

    # =========================================================================
    # 🧬 МОДУЛЬ ДОФАМИНОВОЙ ПАМЯТИ И НАКАЗАНИЙ «БИТЬ ТОКОМ»
    # =========================================================================
    # Память: храним топ-3 лучших схождения за всю историю обучения
    best_matches_memory = [0.0, 0.0, 0.0] 

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
    print("\n==================================================================")
    print("🌍 АДАПТИВНАЯ СИСТЕМА ИНВЕРСИИ НЕПАЛ-2026 С БАЛАНСОМ ГРАДИЕНТОВ")
    print("==================================================================")

    try:
        for epoch in range(8001):
            optimizer.zero_grad()
            
            x_pts = torch.rand(200, 1, requires_grad=True) * (W_PIXELS * DX_METERS)
            y_pts = torch.rand(200, 1, requires_grad=True) * (H_PIXELS * DX_METERS)
            t_pts = torch.rand(200, 1, requires_grad=True) * 10.0 
            
            loss_p, h_w_predicted = compute_nepal_physics_loss(
                pinn, parser, x_pts, y_pts, t_pts, 
                tau_y_guess, K_guess, melt_guess
            )
            
            sat_real_track = torch.where(y_pts < (H_PIXELS * DX_METERS * 0.4), torch.ones_like(y_pts), torch.zeros_like(y_pts))
            sim_track_binary = torch.sigmoid((h_w_predicted - 0.1) * 20.0) 
            loss_satellite = torch.mean((sim_track_binary - sat_real_track)**2)
            
            t0 = torch.zeros_like(x_pts)
            h_ice_t0_pred = torch.exp(pinn(x_pts, y_pts, t0)[:, 1:2])
            h_ice_t0_real = torch.where(y_pts > (H_PIXELS * DX_METERS * 0.8), torch.full_like(y_pts, 30.0), torch.zeros_like(y_pts))
            loss_boundary = torch.mean((h_ice_t0_pred - h_ice_t0_real)**2)

            # Взвешенный баланс
            # total_loss = lambda_p * loss_p + lambda_sat * loss_satellite + loss_boundary

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

            # --- 🧬 МОЗГОВОЙ ЦЕНТР: ПРИНЯТИЕ РЕШЕНИЙ ИИ ---
            with torch.no_grad():
                current_match = ((sim_track_binary > 0.5) == (sat_real_track > 0.5)).float().mean().item() * 100.0

            # 1. Проверяем память топ-3
            if current_match > min(best_matches_memory) and current_match > 60.0:
                # Убираем худшее из топ-3, вставляем новое, сортируем по возрастанию
                best_matches_memory[0] = current_match
                best_matches_memory.sort()
                
                # 🎉 ЭФФЕКТ ДОФАМИНА (Поощрение за прорыв)
                # ИИ нащупал верный путь! Выделяем дофамин: увеличиваем шаг обучения, чтобы быстрее бежать вперед
                dopamine_level = min(3.0, dopamine_level + 0.5)
                shock_voltage = max(0.0, shock_voltage - 50.0) # Снимаем стресс
                
                # Применяем дофамин к оптимизатору
                for param_group in optimizer.param_groups:
                    param_group['lr'] = BASE_LR * dopamine_level
                    
                print(f"🧠 [ДОФАМИН х{dopamine_level:.1f}] Новое топ-схождение: {current_match:.2f}%! Память топ-3: {best_matches_memory}")

            # 2. Наказание «УДАР ТОКОМ» за жесткую деградацию
            # Если схождение упало сильно ниже лучшего результата из памяти
            elif max(best_matches_memory) > 0 and current_match < (max(best_matches_memory) - 5.5):
                # ⚡ БЬЕМ ТОКОМ (Жесткий штраф)
                # Включаем электрический стул: резко врубаем shock_voltage и снижаем шаг обучения, чтобы ИИ «замер» и перестал тупить
                # Тормозим шаг мягче (вычитаем 0.15 вместо 0.3, чтобы не парализовать сеть)
                dopamine_level = max(0.2, dopamine_level - 0.15)
                # Напряжение копится плавнее (+20.0V вместо +80.0V), потолок зажали на 300V
                shock_voltage = min(300.0, shock_voltage + 20.0)  # shock_voltage += 100.0 # Навешиваем огромный штраф в лосс на следующую эпоху
                
                for param_group in optimizer.param_groups:
                    param_group['lr'] = BASE_LR * dopamine_level

                    # 🛡️ ЖЕСТКАЯ ЗАПЛАТКА: ОПЕРАЦИЯ ROLLBACK (Возврат к рекорду)
                    # Если ток зашкаливает выше 300V, значит ИИ застрял в деградации. 
                    # Насильно возвращаем его на истинный путь из памяти Джулии!
                    if shock_voltage > 300.0 and os.path.exists("landslide_pinn_best.pth"):
                        if epoch % 40 == 0:    
                            # Сделали принт честным и динамическим!
                            print(f"🧠 [КРИЗИСНЫЙ ЦЕНТР ИИ] Текущий путь ошибочен! Делаем откат весов к рекорду {max(best_matches_memory):.2f}%...")
                        
                        checkpoint = torch.load("landslide_pinn_best.pth", map_location=torch.device('cpu'), weights_only=True)
                        pinn.load_state_dict(checkpoint)
                        
                        # Сбрасываем стресс и возвращаем нормальный дофамин
                        shock_voltage = 0.0
                        dopamine_level = 1.0   

                # 🔥 ЗАПЛАТКА ОТ СПАМА: выводим принт реже
                if epoch % 100 == 0:
                    print(f"⚡ [БЬЕМ ТОКОМ! Ток: {shock_voltage}V] Схождение просело до {current_match:.2f}%! Замораживаем шаг ИИ до x{dopamine_level:.1f}")

            # Медленное угасание дофамина со временем (ИИ успокаивается)
            dopamine_level = max(1.0, dopamine_level - 0.005)
            shock_voltage = max(0.0, shock_voltage - 0.5)
            
            # Интегрируем электрический штраф в лосс на следующую эпоху см лосс выше

            # 🔁 Адаптивная корректировка весов каждые 10 эпох
            if epoch % 10 == 0:
                with torch.no_grad():
                    # Масштабируем коэффициенты, чтобы уравнять влияние физики и маски спутника
                    ratio = loss_p.item() / (loss_satellite.item() + 1e-6)
                    if ratio > 0:
                        lambda_sat = min(500.0, max(10.0, ratio * 0.1))

            # === ВСТАВИТЬ ПОСЛЕ optimizer.step() ===
            if epoch % 250 == 0:  # Печатаем каждые 50 эпох
                print(f"\n📊 --- Баланс сил на эпохе {epoch} ---")
                print(f"  🧮 Общий Loss:        {total_loss.item():.4f}")
                print(f"  ⚙️ Вес физики:        {loss_p.item():.4f}")
                print(f"  🛰️ Вес спутника (x5):  {(loss_satellite * 5.0).item():.4f}")
                print(f"  🏔️ Вес границ (t=0):  {loss_boundary.item():.4f}")

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
     