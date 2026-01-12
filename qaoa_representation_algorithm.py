
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# --- CONFIGURACIÓN DE DATOS ---
steps = np.array([0, 1, 2, 3, 4])
labels = ['H (Start)', 'C (Cost)', 'B (Mixer)', 'C (Cost)', 'B (Mixer)']

# Datos
y_prob = np.array([0.15, 0.15, 0.45, 0.42, 0.85]) 
y_phase = np.array([0.00, 0.60, 0.25, 0.65, 0.20]) 

# Colores
COLOR_PROB = '#1f77b4' # Azul
COLOR_PHASE = '#d62728' # Rojo

# --- CONFIGURACIÓN ESTÉTICA ---
fig, ax1 = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('white') 
ax1.set_facecolor('white')

ax2 = ax1.twinx()

# Márgenes
fig.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.15)

# --- Eje X ---
ax1.set_xlim(-0.5, 4.5)
ax1.set_xticks(steps)
ax1.set_xticklabels(labels, fontsize=12, color='black')
ax1.annotate('', xy=(4.5, 0), xytext=(-0.5, 0),
            arrowprops=dict(arrowstyle="->", color='black', lw=1.5))

# --- Eje Y Izquierdo (Fase) ---
ax1.set_ylim(0, 1.0)
ax1.set_ylabel("Relative Phase (Arbitrary Units)", color=COLOR_PHASE, fontsize=14, fontweight='bold')
ax1.tick_params(axis='y', labelcolor=COLOR_PHASE, colors=COLOR_PHASE)
ax1.spines['top'].set_visible(False)
ax1.spines['bottom'].set_color('black')
ax1.spines['left'].set_color(COLOR_PHASE)
ax1.spines['left'].set_linewidth(2)
ax1.spines['right'].set_visible(False)

# --- Eje Y Derecho (Probabilidad) ---
ax2.set_ylim(0, 1.0)
ax2.set_ylabel("Probability P(x)", color=COLOR_PROB, fontsize=14, rotation=270, labelpad=20, fontweight='bold')
ax2.tick_params(axis='y', labelcolor=COLOR_PROB, colors=COLOR_PROB)
ax2.spines['top'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.spines['right'].set_color(COLOR_PROB)
ax2.spines['right'].set_linewidth(2)

# --- ELEMENTOS ---
line_phase, = ax1.plot([], [], color=COLOR_PHASE, lw=2, linestyle='--', alpha=0.7, label='Other States (Phase measure)')
dots_phase, = ax1.plot([], [], 'o', color=COLOR_PHASE, markersize=6, alpha=0.7)

line_prob, = ax2.plot([], [], color=COLOR_PROB, lw=3, label='Optimal Solution (Probability)')
dots_prob, = ax2.plot([], [], 'o', color=COLOR_PROB, markersize=8)

status_text = ax1.text(0.5, 0.95, '', transform=ax1.transAxes, ha='center', color='#333333', fontsize=14, fontweight='bold')

# --- VELOCIDAD ---
frames_per_step = 60  
total_frames = (len(steps) - 1) * frames_per_step

def init():
    line_prob.set_data([], [])
    line_phase.set_data([], [])
    dots_prob.set_data([], [])
    dots_phase.set_data([], [])
    status_text.set_text('')
    return line_prob, line_phase, dots_prob, dots_phase, status_text

def update(frame):
    segment = frame // frames_per_step
    progress = (frame % frames_per_step) / frames_per_step
    
    if segment >= len(steps) - 1:
        return line_prob, line_phase, dots_prob, dots_phase, status_text

    current_x = np.linspace(0, segment + progress, frame + 1)
    
    all_y_prob = np.interp(current_x, steps, y_prob)
    all_y_phase = np.interp(current_x, steps, y_phase)

    line_prob.set_data(current_x, all_y_prob)
    line_phase.set_data(current_x, all_y_phase)

    passed_indices = np.where(steps <= (segment + progress))[0]
    dots_prob.set_data(steps[passed_indices], y_prob[passed_indices])
    dots_phase.set_data(steps[passed_indices], y_phase[passed_indices])

    if segment == 0:
        status_text.set_text("1. Cost Hamiltonian (C): Applying phases...")
    elif segment == 1:
        status_text.set_text("2. Mixer Hamiltonian (B): Interference increases probability...")
    elif segment == 2:
        status_text.set_text("3. Cost Hamiltonian (C): Refining phases again...")
    elif segment == 3:
        status_text.set_text("4. Mixer Hamiltonian (B): Final probability amplification!")

    return line_prob, line_phase, dots_prob, dots_phase, status_text

ani = FuncAnimation(fig, update, frames=total_frames + 50, init_func=init, blit=True)

# --- LEYENDA AJUSTADA (BBOX_TO_ANCHOR) ---
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()

# CAMBIO AQUÍ: bbox_to_anchor=(0, 0.88) baja la leyenda un poco (el 1.0 sería el tope)
ax1.legend(h1+h2, l1+l2, loc='upper left', bbox_to_anchor=(0, 0.88), frameon=True, facecolor='white', framealpha=1)

plt.title("QAOA: Phase vs. Probability Evolution", color='black', fontsize=16, pad=20)

# --- GUARDAR ---
print("Generando video (Leyenda reubicada)...")
try:
    writer = FFMpegWriter(fps=30, metadata=dict(artist='Me'), bitrate=2500)
    ani.save("qaoa_legend_fixed.mp4", writer=writer)
    print("¡Listo! Video guardado como 'qaoa_legend_fixed.mp4'")
except Exception as e:
    print(f"Error MP4: {e}. Guardando GIF...")
    ani.save("qaoa_legend_fixed.gif", writer='pillow', fps=20)
    print("¡Listo! GIF guardado.")
