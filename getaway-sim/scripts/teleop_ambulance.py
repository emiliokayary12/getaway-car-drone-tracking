"""
Teleoperación de la ambulancia via flechas del teclado (curses).

Comportamiento:
  - Lanzar `make teleop` solo: NO toma control. Auto-circle sigue corriendo.
  - Mantener una flecha presionada: TOMA control mientras la tecla está activa
    (touch /tmp/manual_mode, publica Twist). Auto-circle se pausa.
  - Soltar la tecla: libera control (rm /tmp/manual_mode, deja de publicar).
    Auto-circle retoma en el siguiente pulso.
  - Press Q: sale, restaura auto definitivamente.

Esto da la sensación de un override momentáneo tipo videojuego — el auto
es el default y el manual solo interviene mientras hay input real.

Controles:
    ↑   adelante
    ↓   reversa
    ←   girar izquierda
    →   girar derecha
    espacio  frenar y mantener
    Q   salir
"""

import curses
import pathlib
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

MANUAL_FLAG  = pathlib.Path("/tmp/manual_mode")
# Topic name idéntico al lado GZ — ros_gz_bridge no remapea el lado GZ.
TOPIC_ROS    = "/model/ambulance/cmd_vel"

V_FWD        = 5.0    # m/s al presionar ↑
V_REV        = -3.0   # m/s al presionar ↓
W_TURN       = 0.8    # rad/s al presionar ← / →
KEY_TIMEOUT  = 0.18   # s sin keystroke → suelta (autorepeat ≈30Hz)
LOOP_HZ      = 20.0


class TeleopNode(Node):
    def __init__(self):
        super().__init__("ambulance_teleop")
        self.pub = self.create_publisher(Twist, TOPIC_ROS, 10)

    def publish(self, vx: float, wz: float) -> None:
        msg = Twist()
        msg.linear.x = vx
        msg.angular.z = wz
        self.pub.publish(msg)


def main(stdscr) -> None:
    rclpy.init()
    node = TeleopNode()

    # NO tocar el flag aquí — auto debe seguir corriendo hasta primer input.

    stdscr.nodelay(True)
    stdscr.timeout(int(1000.0 / LOOP_HZ))
    curses.curs_set(0)

    stdscr.addstr(0, 0, "AMBULANCE TELEOP -- arrows: drive | space: brake | Q: quit")
    stdscr.addstr(1, 0, "Status: AUTO active (no input yet)                  ")
    stdscr.refresh()

    vx, wz = 0.0, 0.0
    last_key = 0.0
    manual_owned = False  # ¿somos los dueños del flag actualmente?

    try:
        while True:
            c = stdscr.getch()
            now = time.time()

            if c == curses.KEY_UP:
                vx, last_key = V_FWD, now
            elif c == curses.KEY_DOWN:
                vx, last_key = V_REV, now
            elif c == curses.KEY_LEFT:
                wz, last_key = W_TURN, now
            elif c == curses.KEY_RIGHT:
                wz, last_key = -W_TURN, now
            elif c == ord(' '):
                # Espacio = freno mantenido. Mantiene control hasta soltar.
                vx, wz = 0.0, 0.0
                last_key = now
            elif c in (ord('q'), ord('Q')):
                break

            key_active = (now - last_key) <= KEY_TIMEOUT

            if key_active:
                # Hay input → tomar control si no lo teníamos
                if not manual_owned:
                    MANUAL_FLAG.touch()
                    manual_owned = True
                    stdscr.addstr(1, 0, "Status: MANUAL active (auto paused)              ")
                node.publish(vx, wz)
            else:
                # Sin input → soltar control si lo teníamos
                if manual_owned:
                    node.publish(0.0, 0.0)  # un pulso a cero para detener motion
                    time.sleep(0.05)
                    node.publish(0.0, 0.0)
                    try:
                        MANUAL_FLAG.unlink()
                    except FileNotFoundError:
                        pass
                    manual_owned = False
                    vx, wz = 0.0, 0.0
                    stdscr.addstr(1, 0, "Status: AUTO active (released)                   ")
                # No publica nada — los pulsos del auto-loop llegan sin estorbo

            stdscr.addstr(3, 0, f"vx = {vx:+.2f} m/s   wz = {wz:+.2f} rad/s   ")
            stdscr.refresh()
    finally:
        # Salida limpia: cero velocidad + flag fuera
        try:
            node.publish(0.0, 0.0)
            time.sleep(0.05)
            node.publish(0.0, 0.0)
        except Exception:
            pass
        try:
            MANUAL_FLAG.unlink()
        except FileNotFoundError:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    curses.wrapper(main)
