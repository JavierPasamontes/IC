import pygame
import heapq

# Configuración de la cuadrícula
WIDTH, HEIGHT = 500, 500
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

# Inicializar Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("A* Pathfinding Visualization")

# Clase Nodo
class Node:
    def __lt__(self, other):
        return False  # No es necesario comparar nodos directamente


    def __init__(self, row, col, penalty=1):
        self.row = row
        self.col = col
        self.penalty = penalty
        self.color = WHITE
        self.neighbors = []
    
    def draw(self):
        pygame.draw.rect(screen, self.color, (self.col * CELL_SIZE, self.row * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        pygame.draw.rect(screen, GRAY, (self.col * CELL_SIZE, self.row * CELL_SIZE, CELL_SIZE, CELL_SIZE), 1)
    
    def update_neighbors(self, grid):
        self.neighbors = []
        directions = [
            (0, 1), (1, 0), (0, -1), (-1, 0),  # Movimiento horizontal y vertical
            (1, 1), (-1, -1), (1, -1), (-1, 1)  # Movimiento diagonal
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

# Crear cuadrícula
grid = [[Node(r, c) for c in range(COLS)] for r in range(ROWS)]
for row in grid:
    for node in row:
        node.update_neighbors(grid)

start, end = None, None
running = True
while running:
    screen.fill(WHITE)
    for row in grid:
        for node in row:
            node.draw()
    pygame.display.flip()
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        
        if pygame.mouse.get_pressed()[0]:  # Clic izquierdo
            x, y = pygame.mouse.get_pos()
            row, col = y // CELL_SIZE, x // CELL_SIZE
            node = grid[row][col]
            if not start:
                start = node
                start.color = GREEN
            elif not end and node != start:
                end = node
                end.color = RED
            elif node != start and node != end:
                node.color = BLACK
        
        if pygame.mouse.get_pressed()[2]:  # Clic derecho (celdas con penalización)
            x, y = pygame.mouse.get_pos()
            row, col = y // CELL_SIZE, x // CELL_SIZE
            node = grid[row][col]
            if node != start and node != end:
                node.color = BLUE
                node.penalty = 5
        
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE and start and end:
                for row in grid:
                    for node in row:
                        node.update_neighbors(grid)
                a_star(grid, start, end)

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_e:
                pygame.quit()

pygame.quit()
