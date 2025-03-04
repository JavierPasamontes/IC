import pygame
import heapq

# Configuración de la cuadrícula
WIDTH, HEIGHT = 500, 580  # Espacio extra para la leyenda
ROWS, COLS = 20, 20
CELL_SIZE = WIDTH // COLS

# Colores
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
GRAY = (200, 200, 200)
DARK_GRAY = (50, 50, 50)  # Fondo de la leyenda

# Inicializar Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("A* Pathfinding Visualization")
font = pygame.font.Font(None, 18)

# Variables para modo de edición
placing_mode = None
start, end = None, None

# Función para mostrar la leyenda en dos columnas
def draw_legend():
    pygame.draw.rect(screen, DARK_GRAY, (0, 0, WIDTH, 80))  # Fondo de la leyenda

    # Leyenda dividida en dos columnas
    legend_items_left = [
        ("A", GREEN, "Poner inicio"),
        ("D", RED, "Poner destino"),
        ("S", BLACK, "Poner obstáculo"),
        ("W", BLUE, "Poner waypoint")
    ]

    legend_items_right = [
        ("Clic derecho", WHITE, "Borrar casilla"),
        ("Espacio", WHITE, "Ejecutar A*"),
        ("R", WHITE, "Reiniciar cuadrícula"),
        ("E", WHITE, "Salir")
    ]

    # Dibujar la columna izquierda
    for i, (key, color, desc) in enumerate(legend_items_left):
        key_label = font.render(key, True, color)
        desc_label = font.render(f": {desc}", True, WHITE)
        screen.blit(key_label, (10, 10 + i * 18))
        screen.blit(desc_label, (30, 10 + i * 18))

    # Dibujar la columna derecha
    for i, (key, color, desc) in enumerate(legend_items_right):
        key_label = font.render(key, True, color)
        desc_label = font.render(f": {desc}", True, WHITE)
        screen.blit(key_label, (WIDTH // 2 + 30, 10 + i * 18))
        screen.blit(desc_label, (WIDTH // 2 + 120, 10 + i * 18))

# Clase Nodo
class Node:
    def __lt__(self, other):
        return False

    def __init__(self, row, col, penalty=1):
        self.row = row
        self.col = col
        self.penalty = penalty
        self.color = WHITE
        self.neighbors = []

    def draw(self):
        pygame.draw.rect(screen, self.color, (self.col * CELL_SIZE, self.row * CELL_SIZE + 80, CELL_SIZE, CELL_SIZE))
        pygame.draw.rect(screen, GRAY, (self.col * CELL_SIZE, self.row * CELL_SIZE + 80, CELL_SIZE, CELL_SIZE), 1)

    def update_neighbors(self, grid):
        self.neighbors = []
        directions = [
            (0, 1), (1, 0), (0, -1), (-1, 0),
            (1, 1), (-1, -1), (1, -1), (-1, 1)
        ]
        for dr, dc in directions:
            r, c = self.row + dr, self.col + dc
            if 0 <= r < ROWS and 0 <= c < COLS and grid[r][c].color != BLACK:
                self.neighbors.append(grid[r][c])

# Algoritmo A*
def heuristic(a, b):
    return abs(a.row - b.row) + abs(a.col - b.col)

def a_star(grid, start, end):
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {node: float('inf') for row in grid for node in row}
    g_score[start] = 0
    f_score = {node: float('inf') for row in grid for node in row}
    f_score[start] = heuristic(start, end)

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == end:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            for node in path:
                node.color = YELLOW
            
            start.color = GREEN
            end.color = RED
            return

        for neighbor in current.neighbors:
            temp_g_score = g_score[current] + neighbor.penalty
            if temp_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = temp_g_score
                f_score[neighbor] = temp_g_score + heuristic(neighbor, end)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

# Función para reiniciar la cuadrícula
def reset_grid():
    global grid, start, end
    grid = [[Node(r, c) for c in range(COLS)] for r in range(ROWS)]
    for row in grid:
        for node in row:
            node.update_neighbors(grid)
    start, end = None, None

# Crear cuadrícula inicial
reset_grid()

running = True

while running:
    screen.fill(WHITE)
    draw_legend()

    for row in grid:
        for node in row:
            node.draw()
    
    pygame.display.flip()
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        
        if pygame.mouse.get_pressed()[0]:  # Clic izquierdo
            x, y = pygame.mouse.get_pos()
            if y > 80:  # Evita la zona de la leyenda
                row, col = (y - 80) // CELL_SIZE, x // CELL_SIZE
                node = grid[row][col]
                
                if placing_mode == "start":
                    if start:
                        start.color = WHITE  # Borra el anterior
                    start = node
                    start.color = GREEN

                elif placing_mode == "end":
                    if end:
                        end.color = WHITE  # Borra el anterior
                    end = node
                    end.color = RED

                elif placing_mode == "obstacle":
                    node.color = BLACK

                elif placing_mode == "waypoint":
                    node.color = BLUE
                    node.penalty = 5  # Penalización extra
        
        if pygame.mouse.get_pressed()[2]:  # Clic derecho (BORRAR)
            x, y = pygame.mouse.get_pos()
            if y > 80:
                row, col = (y - 80) // CELL_SIZE, x // CELL_SIZE
                node = grid[row][col]
                node.color = WHITE  # Borra cualquier color

                if node == start:
                    start = None
                elif node == end:
                    end = None
        
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_a:
                placing_mode = "start"
            elif event.key == pygame.K_d:
                placing_mode = "end"
            elif event.key == pygame.K_s:
                placing_mode = "obstacle"
            elif event.key == pygame.K_w:
                placing_mode = "waypoint"
            elif event.key == pygame.K_r:
                reset_grid()  # Reiniciar cuadrícula
            elif event.key == pygame.K_SPACE and start and end:
                for row in grid:
                    for node in row:
                        node.update_neighbors(grid)
                a_star(grid, start, end)
            elif event.key == pygame.K_e:
                pygame.quit()

pygame.quit()