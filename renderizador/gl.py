#!/usr/bin/env python3
# -*- coding: UTF-8 -*-


"""
Biblioteca Gráfica / Graphics Library.

Desenvolvido por: <SEU NOME AQUI>
Disciplina: Computação Gráfica
Data: <DATA DE INÍCIO DA IMPLEMENTAÇÃO>
"""

import math  # Funções matemáticas
import time  # Para operações com tempo

import gpu  # Simula os recursos de uma GPU
import numpy as np  # Biblioteca do Numpy

width = 300
height = 200


class GL:
    transformation_stack = [np.identity(4)]
    view_matrix = np.identity(4)
    projection_matrix = np.identity(4)
    """Classe que representa a biblioteca gráfica (Graphics Library)."""

    width = 800  # largura da tela
    height = 600  # altura da tela
    near = 0.01  # plano de corte próximo
    far = 1000  # plano de corte distante

    @staticmethod
    def setup(width, height, near=0.01, far=1000):
        """Definr parametros para câmera de razão de aspecto, plano próximo e distante."""
        GL.width = width
        GL.height = height
        GL.near = near
        GL.far = far

    @staticmethod
    def draw_point_safe(x, y, color):
        """Desenha um ponto na tela se estiver dentro dos limites."""
        if 0 <= x < GL.width and 0 <= y < GL.height:
            gpu.GPU.draw_pixel([x, y], gpu.GPU.RGB8, color)

    @staticmethod
    def polypoint2D(
        point: list[float], colors: dict[str, list[float]]
    ) -> None:
        """Função usada para renderizar Polypoint2D."""
        emissive_color = colors.get("emissiveColor", [1, 1, 1])
        color_255 = [int(c * 255) for c in emissive_color]

        for i in range(0, len(point), 2):
            x = int(point[i])
            y = int(point[i + 1])
            GL.draw_point_safe(x, y, color_255)

    @staticmethod
    def draw_line(x0, y0, x1, y1, color):
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            GL.draw_point_safe(x0, y0, color)
            if x0 == x1 and y0 == y1:
                break
            e2 = err * 2
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

    @staticmethod
    def polyline2D(
        lineSegments: list[float], colors: dict[str, list[float]]
    ) -> None:
        """Função usada para renderizar Polyline2D."""
        points = [
            (lineSegments[i], lineSegments[i + 1])
            for i in range(0, len(lineSegments), 2)
        ]

        emissive_color = colors.get("emissiveColor", [1, 1, 1])
        color = [int(c * 255) for c in emissive_color]

        for i in range(len(points) - 1):
            x0, y0 = points[i]
            x1, y1 = points[i + 1]
            GL.draw_line(int(x0), int(y0), int(x1), int(y1), color)

    @staticmethod
    def triangleSet2D(vertices, colors):
        """Função usada para renderizar TriangleSet2D."""
        emissive_color = colors.get("emissiveColor", [1, 1, 1])
        color = [int(c * 255) for c in emissive_color]

        diffuse_color = colors.get("diffuseColor", [1, 1, 1])
        if color == [0, 0, 0]:
            color = [int(c * 255) for c in diffuse_color]

        def determinant(xa, ya, xb, yb):
            return xa * yb - ya * xb

        def L(x, y, x0, y0, x1, y1):
            return determinant(x - x0, y - y0, x1 - x0, y1 - y0)

        print("OOOOOOOOOOOOOOOO", vertices)
        for i in range(0, len(vertices), 6):
            x0, y0 = vertices[i] + 0.5, vertices[i + 1] + 0.5
            x1, y1 = vertices[i + 2] + 0.5, vertices[i + 3] + 0.5
            x2, y2 = vertices[i + 4] + 0.5, vertices[i + 5] + 0.5

            min_x = min(x0, x1, x2)
            max_x = max(x0, x1, x2)
            min_y = min(y0, y1, y2)
            max_y = max(y0, y1, y2)

            for sx in range(int(min_x), int(max_x) + 1):
                for sy in range(int(min_y), int(max_y) + 1):
                    L0 = L(sx, sy, x0, y0, x1, y1)
                    L1 = L(sx, sy, x1, y1, x2, y2)
                    L2 = L(sx, sy, x2, y2, x0, y0)

                    if L0 >= 0 and L1 >= 0 and L2 >= 0:
                        GL.draw_point_safe(sx, sy, color)

    @staticmethod
    def circle2D(radius, colors):
        """Função usada para renderizar Circle2D."""
        emissive_color = colors.get("emissiveColor", [1, 1, 1])
        color = [int(c * 255) for c in emissive_color]

        center_x, center_y = 0, 0

        min_x = int(center_x - radius)
        max_x = int(center_x + radius)
        min_y = int(center_y - radius)
        max_y = int(center_y + radius)

        for sx in range(min_x, max_x + 1):
            for sy in range(min_y, max_y + 1):
                if (sx - center_x) ** 2 + (sy - center_y) ** 2 <= radius**2:
                    GL.draw_point_safe(sx, sy, color)

    @staticmethod
    def triangleSet(point, colors):
        """Função usada para renderizar TriangleSet."""
        # Nessa função você receberá pontos no parâmetro point, esses pontos são uma lista
        # de pontos x, y, e z sempre na ordem. Assim point[0] é o valor da coordenada x do
        # primeiro ponto, point[1] o valor y do primeiro ponto, point[2] o valor z da
        # coordenada z do primeiro ponto. Já point[3] é a coordenada x do segundo ponto e
        # assim por diante.
        # No TriangleSet os triângulos são informados individualmente, assim os três
        # primeiros pontos definem um triângulo, os três próximos pontos definem um novo
        # triângulo, e assim por diante.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, você pode assumir
        # inicialmente, para o TriangleSet, o desenho das linhas com a cor emissiva
        # (emissiveColor), conforme implementar novos materias você deverá suportar outros
        # tipos de cores.
        for i in range(0, len(point), 9):
            vertices = np.array(
                [
                    [point[i], point[i + 3], point[i + 6]],
                    [point[i+1], point[i + 4], point[i + 7]],
                    [point[i+2], point[i + 5], point[i + 8]],
                    [1, 1, 1],
                ]
            )

            for i in range(len(GL.transformation_stack) - 1, -1, -1):
                vertices = np.matmul(GL.transformation_stack[i], vertices)

            matrix_look_at = np.matmul(GL.view_matrix, vertices)
            NDC = np.matmul(GL.projection_matrix, matrix_look_at)

            # normalize vertices
            p0 = np.array(
                [
                    NDC[0][0] / NDC[3][0],
                    NDC[1][0] / NDC[3][0],
                    NDC[2][0] / NDC[3][0],
                ]
            )
            p1 = np.array(
                [
                    NDC[0][1] / NDC[3][1],
                    NDC[1][1] / NDC[3][1],
                    NDC[2][1] / NDC[3][1],
                ]
            )
            p2 = np.array(
                [
                    NDC[0][2] / NDC[3][2],
                    NDC[1][2] / NDC[3][2],
                    NDC[2][2] / NDC[3][2],
                ]
            )
                
            width = 300
            height = 200
            tela_matrix = np.array(
                [
                    [width / 2, 0, 0, width / 2],
                    [0, -height / 2, 0, height / 2],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ]
            )
            vertices_tela = np.array(
                [
                    [p0[0], p1[0], p2[0]],
                    [p0[1], p1[1], p2[1]],
                    [p0[2], p1[2], p2[2]],
                    [1, 1, 1],
                ]
            )

            vertices_final = np.matmul(tela_matrix, vertices_tela)

            pontos = [
                vertices_final[0][0],
                vertices_final[1][0],
                vertices_final[0][1],
                vertices_final[1][1],
                vertices_final[0][2],
                vertices_final[1][2],
            ]

            pontos = np.array(pontos)
            GL.triangleSet2D(pontos, colors)

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print(
            "TriangleSet : pontos = {0}".format(point)
        )  # imprime no terminal pontos
        print(
            "TriangleSet : colors = {0}".format(colors)
        )  # imprime no terminal as cores

        # Exemplo de desenho de um pixel branco na coordenada 10, 10
        gpu.GPU.draw_pixel(
            [10, 10], gpu.GPU.RGB8, [255, 255, 255]
        )  # altevertices[i + 2]ra pixel

    @staticmethod
    def viewpoint(position, orientation, fieldOfView):
        """Função usada para renderizar (na verdade coletar os dados) de Viewpoint."""
        # Na função de viewpoint você receberá a posição, orientação e campo de visão da
        # câmera virtual. Use esses dados para poder calcular e criar a matriz de projeção
        # perspectiva para poder aplicar nos pontos dos objetos geométricos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        matrix_translation = np.array(
            [
                [1, 0, 0, position[0]],
                [0, 1, 0, position[1]],
                [0, 0, 1, position[2]],
                [0, 0, 0, 1],
            ]
        )
        x, y, z, a = orientation
        normalize = np.linalg.norm([x, y, z])
        x /= normalize
        y /= normalize
        z /= normalize

        half_angle = a / 2
        w = np.cos(half_angle)
        sin_half_angle = np.sin(half_angle)
        x *= sin_half_angle
        y *= sin_half_angle
        z *= sin_half_angle
        matrix_rotation = np.array(
            [
                [
                    1 - 2 * y**2 - 2 * z**2,
                    2 * x * y - 2 * w * z,
                    2 * x * z + 2 * w * y,
                    0,
                ],
                [
                    2 * x * y + 2 * w * z,
                    1 - 2 * x**2 - 2 * z**2,
                    2 * y * z - 2 * w * x,
                    0,
                ],
                [
                    2 * x * z - 2 * w * y,
                    2 * y * z + 2 * w * x,
                    1 - 2 * x**2 - 2 * y**2,
                    0,
                ],
                [0, 0, 0, 1],
            ]
        )

        # invert matrix
        matrix_translation = np.linalg.inv(matrix_translation)
        matrix_rotation = np.linalg.inv(matrix_rotation)

        GL.view_matrix = np.matmul(matrix_rotation, matrix_translation)

        fovy = 2 * np.arctan(
            np.tan(fieldOfView / 2)
            * (height / ((width**2 + height**2) ** 0.5))
        )

        aspect_ratio = width / height
        near = 0.01
        far = 1000
        top = near * np.tan(fovy)
        right = top * aspect_ratio

        GL.projection_matrix = np.array(
            [
                [near / right, 0, 0, 0],
                [0, near / top, 0, 0],
                [0, 0, -((far + near) / (far - near)), (-2 * far * near) / (far - near)],
                [0, 0, -1, 0],
            ]
        )

        print("Viewpoint : ", end="")
        print("position = {0} ".format(position), end="")
        print("orientation = {0} ".format(orientation), end="")
        print("fieldOfView = {0} ".format(fieldOfView))

    @staticmethod
    def transform_in(translation, scale, rotation):
        """Função usada para renderizar (na verdade coletar os dados) de Transform."""
        # A função transform_in será chamada quando se entrar em um nó X3D do tipo Transform
        # do grafo de cena. Os valores passados são a escala em um vetor [x, y, z]
        # indicando a escala em cada direção, a translação [x, y, z] nas respectivas
        # coordenadas e finalmente a rotação por [x, y, z, t] sendo definida pela rotação
        # do objeto ao redor do eixo x, y, z por t radianos, seguindo a regra da mão direita.
        # Quando se entrar em um nó transform se deverá salvar a matriz de transformação dos
        # modelos do mundo em alguma estrutura de pilha.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Transform : ", end="")
        # print("translation = {0} ".format(translation), end="")  # imprime no terminal
        matrix_tranform = np.array(
            [
                [1, 0, 0, translation[0]],
                [0, 1, 0, translation[1]],
                [0, 0, 1, translation[2]],
                [0, 0, 0, 1],
            ]
        )
        # print("scale = {0} ".format(scale), end="")  # imprime no terminal
        matrix_scale = np.array(
            [
                [scale[0], 0, 0, 0],
                [0, scale[1], 0, 0],
                [0, 0, scale[2], 0],
                [0, 0, 0, 1],
            ]
        )
        # print("rotation = {0} ".format(rotation), end="")  # imprime no terminal
        x, y, z, t = rotation
        half_angle = t / 2
        w = np.cos(half_angle)
        sin_half_angle = np.sin(half_angle)
        x *= sin_half_angle
        y *= sin_half_angle
        z *= sin_half_angle
        matrix_rotation = np.array(
            [
                [
                    1 - 2 * y**2 - 2 * z**2,
                    2 * x * y - 2 * w * z,
                    2 * x * z + 2 * w * y,
                    0,
                ],
                [
                    2 * x * y + 2 * w * z,
                    1 - 2 * x**2 - 2 * z**2,
                    2 * y * z - 2 * w * x,
                    0,
                ],
                [
                    2 * x * z - 2 * w * y,
                    2 * y * z + 2 * w * x,
                    1 - 2 * x**2 - 2 * y**2,
                    0,
                ],
                [0, 0, 0, 1],
            ]
        )

        matrix_tranform = matrix_tranform.astype(np.float64)
        matrix_rotation = matrix_rotation.astype(np.float64)
        matrix_scale = matrix_scale.astype(np.float64)

        matrix = np.matmul(matrix_tranform, np.matmul(matrix_rotation, matrix_scale))

        GL.transformation_stack.append(matrix)
        print("tranform matrix = ", matrix)

    @staticmethod
    def transform_out():
        """Função usada para renderizar (na verdade coletar os dados) de Transform."""
        # A função transform_out será chamada quando se sair em um nó X3D do tipo Transform do
        # grafo de cena. Não são passados valores, porém quando se sai de um nó transform se
        # deverá recuperar a matriz de transformação dos modelos do mundo da estrutura de
        # pilha implementada.
        GL.transformation_stack.pop()

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Saindo de Transform")

    @staticmethod
    def triangleStripSet(point, stripCount, colors):
        """Função usada para renderizar TriangleStripSet."""
        # A função triangleStripSet é usada para desenhar tiras de triângulos interconectados,
        # você receberá as coordenadas dos pontos no parâmetro point, esses pontos são uma
        # lista de pontos x, y, e z sempre na ordem. Assim point[0] é o valor da coordenada x
        # do primeiro ponto, point[1] o valor y do primeiro ponto, point[2] o valor z da
        # coordenada z do primeiro ponto. Já point[3] é a coordenada x do segundo ponto e assim
        # por diante. No TriangleStripSet a quantidade de vértices a serem usados é informado
        # em uma lista chamada stripCount (perceba que é uma lista). Ligue os vértices na ordem,
        # primeiro triângulo será com os vértices 0, 1 e 2, depois serão os vértices 1, 2 e 3,
        # depois 2, 3 e 4, e assim por diante. Cuidado com a orientação dos vértices, ou seja,
        # todos no sentido horário ou todos no sentido anti-horário, conforme especificado.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("TriangleStripSet : pontos = {0} ".format(point), end="")
        for i, strip in enumerate(stripCount):
            print("strip[{0}] = {1} ".format(i, strip), end="")
        print("")
        print(
            "TriangleStripSet : colors = {0}".format(colors)
        )  # imprime no terminal as cores

        # Exemplo de desenho de um pixel branco na coordenada 10, 10
        gpu.GPU.draw_pixel(
            [10, 10], gpu.GPU.RGB8, [255, 255, 255]
        )  # altera pixel

    @staticmethod
    def indexedTriangleStripSet(point, index, colors):
        """Função usada para renderizar IndexedTriangleStripSet."""
        # A função indexedTriangleStripSet é usada para desenhar tiras de triângulos
        # interconectados, você receberá as coordenadas dos pontos no parâmetro point, esses
        # pontos são uma lista de pontos x, y, e z sempre na ordem. Assim point[0] é o valor
        # da coordenada x do primeiro ponto, point[1] o valor y do primeiro ponto, point[2]
        # o valor z da coordenada z do primeiro ponto. Já point[3] é a coordenada x do
        # segundo ponto e assim por diante. No IndexedTriangleStripSet uma lista informando
        # como conectar os vértices é informada em index, o valor -1 indica que a lista
        # acabou. A ordem de conexão será de 3 em 3 pulando um índice. Por exemplo: o
        # primeiro triângulo será com os vértices 0, 1 e 2, depois serão os vértices 1, 2 e 3,
        # depois 2, 3 e 4, e assim por diante. Cuidado com a orientação dos vértices, ou seja,
        # todos no sentido horário ou todos no sentido anti-horário, conforme especificado.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print(
            "IndexedTriangleStripSet : pontos = {0}, index = {1}".format(
                point, index
            )
        )
        print(
            "IndexedTriangleStripSet : colors = {0}".format(colors)
        )  # imprime as cores

        # Exemplo de desenho de um pixel branco na coordenada 10, 10
        gpu.GPU.draw_pixel(
            [10, 10], gpu.GPU.RGB8, [255, 255, 255]
        )  # altera pixel

    @staticmethod
    def box(size, colors):
        """Função usada para renderizar Boxes."""
        # A função box é usada para desenhar paralelepípedos na cena. O Box é centrada no
        # (0, 0, 0) no sistema de coordenadas local e alinhado com os eixos de coordenadas
        # locais. O argumento size especifica as extensões da caixa ao longo dos eixos X, Y
        # e Z, respectivamente, e cada valor do tamanho deve ser maior que zero. Para desenha
        # essa caixa você vai provavelmente querer tesselar ela em triângulos, para isso
        # encontre os vértices e defina os triângulos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Box : size = {0}".format(size))  # imprime no terminal pontos
        print(
            "Box : colors = {0}".format(colors)
        )  # imprime no terminal as cores

        # Exemplo de desenho de um pixel branco na coordenada 10, 10
        gpu.GPU.draw_pixel(
            [10, 10], gpu.GPU.RGB8, [255, 255, 255]
        )  # altera pixel

    @staticmethod
    def indexedFaceSet(
        coord,
        coordIndex,
        colorPerVertex,
        color,
        colorIndex,
        texCoord,
        texCoordIndex,
        colors,
        current_texture,
    ):
        """Função usada para renderizar IndexedFaceSet."""
        # A função indexedFaceSet é usada para desenhar malhas de triângulos. Ela funciona de
        # forma muito simular a IndexedTriangleStripSet porém com mais recursos.
        # Você receberá as coordenadas dos pontos no parâmetro cord, esses
        # pontos são uma lista de pontos x, y, e z sempre na ordem. Assim coord[0] é o valor
        # da coordenada x do primeiro ponto, coord[1] o valor y do primeiro ponto, coord[2]
        # o valor z da coordenada z do primeiro ponto. Já coord[3] é a coordenada x do
        # segundo ponto e assim por diante. No IndexedFaceSet uma lista de vértices é informada
        # em coordIndex, o valor -1 indica que a lista acabou.
        # A ordem de conexão será de 3 em 3 pulando um índice. Por exemplo: o
        # primeiro triângulo será com os vértices 0, 1 e 2, depois serão os vértices 1, 2 e 3,
        # depois 2, 3 e 4, e assim por diante.
        # Adicionalmente essa implementação do IndexedFace aceita cores por vértices, assim
        # se a flag colorPerVertex estiver habilitada, os vértices também possuirão cores
        # que servem para definir a cor interna dos poligonos, para isso faça um cálculo
        # baricêntrico de que cor deverá ter aquela posição. Da mesma forma se pode definir uma
        # textura para o poligono, para isso, use as coordenadas de textura e depois aplique a
        # cor da textura conforme a posição do mapeamento. Dentro da classe GPU já está
        # implementadado um método para a leitura de imagens.

        # Os prints abaixo são só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("IndexedFaceSet : ")
        if coord:
            print(
                "\tpontos(x, y, z) = {0}, coordIndex = {1}".format(
                    coord, coordIndex
                )
            )
        print("colorPerVertex = {0}".format(colorPerVertex))
        if colorPerVertex and color and colorIndex:
            print(
                "\tcores(r, g, b) = {0}, colorIndex = {1}".format(
                    color, colorIndex
                )
            )
        if texCoord and texCoordIndex:
            print(
                "\tpontos(u, v) = {0}, texCoordIndex = {1}".format(
                    texCoord, texCoordIndex
                )
            )
        if current_texture:
            image = gpu.GPU.load_texture(current_texture[0])
            print("\t Matriz com image = {0}".format(image))
            print("\t Dimensões da image = {0}".format(image.shape))
        print(
            "IndexedFaceSet : colors = {0}".format(colors)
        )  # imprime no terminal as cores

        # Exemplo de desenho de um pixel branco na coordenada 10, 10
        gpu.GPU.draw_pixel(
            [10, 10], gpu.GPU.RGB8, [255, 255, 255]
        )  # altera pixel

    @staticmethod
    def sphere(radius, colors):
        """Função usada para renderizar Esferas."""
        # A função sphere é usada para desenhar esferas na cena. O esfera é centrada no
        # (0, 0, 0) no sistema de coordenadas local. O argumento radius especifica o
        # raio da esfera que está sendo criada. Para desenha essa esfera você vai
        # precisar tesselar ela em triângulos, para isso encontre os vértices e defina
        # os triângulos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print(
            "Sphere : radius = {0}".format(radius)
        )  # imprime no terminal o raio da esfera
        print(
            "Sphere : colors = {0}".format(colors)
        )  # imprime no terminal as cores

    @staticmethod
    def navigationInfo(headlight):
        """Características físicas do avatar do visualizador e do modelo de visualização."""
        # O campo do headlight especifica se um navegador deve acender um luz direcional que
        # sempre aponta na direção que o usuário está olhando. Definir este campo como TRUE
        # faz com que o visualizador forneça sempre uma luz do ponto de vista do usuário.
        # A luz headlight deve ser direcional, ter intensidade = 1, cor = (1 1 1),
        # ambientIntensity = 0,0 e direção = (0 0 −1).

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print(
            "NavigationInfo : headlight = {0}".format(headlight)
        )  # imprime no terminal

    @staticmethod
    def directionalLight(ambientIntensity, color, intensity, direction):
        """Luz direcional ou paralela."""
        # Define uma fonte de luz direcional que ilumina ao longo de raios paralelos
        # em um determinado vetor tridimensional. Possui os campos básicos ambientIntensity,
        # cor, intensidade. O campo de direção especifica o vetor de direção da iluminação
        # que emana da fonte de luz no sistema de coordenadas local. A luz é emitida ao
        # longo de raios paralelos de uma distância infinita.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print(
            "DirectionalLight : ambientIntensity = {0}".format(
                ambientIntensity
            )
        )
        print(
            "DirectionalLight : color = {0}".format(color)
        )  # imprime no terminal
        print(
            "DirectionalLight : intensity = {0}".format(intensity)
        )  # imprime no terminal
        print(
            "DirectionalLight : direction = {0}".format(direction)
        )  # imprime no terminal

    @staticmethod
    def pointLight(ambientIntensity, color, intensity, location):
        """Luz pontual."""
        # Fonte de luz pontual em um local 3D no sistema de coordenadas local. Uma fonte
        # de luz pontual emite luz igualmente em todas as direções; ou seja, é omnidirecional.
        # Possui os campos básicos ambientIntensity, cor, intensidade. Um nó PointLight ilumina
        # a geometria em um raio de sua localização. O campo do raio deve ser maior ou igual a
        # zero. A iluminação do nó PointLight diminui com a distância especificada.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("PointLight : ambientIntensity = {0}".format(ambientIntensity))
        print("PointLight : color = {0}".format(color))  # imprime no terminal
        print(
            "PointLight : intensity = {0}".format(intensity)
        )  # imprime no terminal
        print(
            "PointLight : location = {0}".format(location)
        )  # imprime no terminal

    @staticmethod
    def fog(visibilityRange, color):
        """Névoa."""
        # O nó Fog fornece uma maneira de simular efeitos atmosféricos combinando objetos
        # com a cor especificada pelo campo de cores com base nas distâncias dos
        # vários objetos ao visualizador. A visibilidadeRange especifica a distância no
        # sistema de coordenadas local na qual os objetos são totalmente obscurecidos
        # pela névoa. Os objetos localizados fora de visibilityRange do visualizador são
        # desenhados com uma cor de cor constante. Objetos muito próximos do visualizador
        # são muito pouco misturados com a cor do nevoeiro.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Fog : color = {0}".format(color))  # imprime no terminal
        print("Fog : visibilityRange = {0}".format(visibilityRange))

    @staticmethod
    def timeSensor(cycleInterval, loop):
        """Gera eventos conforme o tempo passa."""
        # Os nós TimeSensor podem ser usados para muitas finalidades, incluindo:
        # Condução de simulações e animações contínuas; Controlar atividades periódicas;
        # iniciar eventos de ocorrência única, como um despertador;
        # Se, no final de um ciclo, o valor do loop for FALSE, a execução é encerrada.
        # Por outro lado, se o loop for TRUE no final de um ciclo, um nó dependente do
        # tempo continua a execução no próximo ciclo. O ciclo de um nó TimeSensor dura
        # cycleInterval segundos. O valor de cycleInterval deve ser maior que zero.

        # Deve retornar a fração de tempo passada em fraction_changed

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print(
            "TimeSensor : cycleInterval = {0}".format(cycleInterval)
        )  # imprime no terminal
        print("TimeSensor : loop = {0}".format(loop))

        # Esse método já está implementado para os alunos como exemplo
        epoch = (
            time.time()
        )  # time in seconds since the epoch as a floating point number.
        fraction_changed = (epoch % cycleInterval) / cycleInterval

        return fraction_changed

    @staticmethod
    def splinePositionInterpolator(set_fraction, key, keyValue, closed):
        """Interpola não linearmente entre uma lista de vetores 3D."""
        # Interpola não linearmente entre uma lista de vetores 3D. O campo keyValue possui
        # uma lista com os valores a serem interpolados, key possui uma lista respectiva de chaves
        # dos valores em keyValue, a fração a ser interpolada vem de set_fraction que varia de
        # zeroa a um. O campo keyValue deve conter exatamente tantos vetores 3D quanto os
        # quadros-chave no key. O campo closed especifica se o interpolador deve tratar a malha
        # como fechada, com uma transições da última chave para a primeira chave. Se os keyValues
        # na primeira e na última chave não forem idênticos, o campo closed será ignorado.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print(
            "SplinePositionInterpolator : set_fraction = {0}".format(
                set_fraction
            )
        )
        print(
            "SplinePositionInterpolator : key = {0}".format(key)
        )  # imprime no terminal
        print("SplinePositionInterpolator : keyValue = {0}".format(keyValue))
        print("SplinePositionInterpolator : closed = {0}".format(closed))

        # Abaixo está só um exemplo de como os dados podem ser calculados e transferidos
        value_changed = [0.0, 0.0, 0.0]

        return value_changed

    @staticmethod
    def orientationInterpolator(set_fraction, key, keyValue):
        """Interpola entre uma lista de valores de rotação especificos."""
        # Interpola rotações são absolutas no espaço do objeto e, portanto, não são cumulativas.
        # Uma orientação representa a posição final de um objeto após a aplicação de uma rotação.
        # Um OrientationInterpolator interpola entre duas orientações calculando o caminho mais
        # curto na esfera unitária entre as duas orientações. A interpolação é linear em
        # comprimento de arco ao longo deste caminho. Os resultados são indefinidos se as duas
        # orientações forem diagonalmente opostas. O campo keyValue possui uma lista com os
        # valores a serem interpolados, key possui uma lista respectiva de chaves
        # dos valores em keyValue, a fração a ser interpolada vem de set_fraction que varia de
        # zeroa a um. O campo keyValue deve conter exatamente tantas rotações 3D quanto os
        # quadros-chave no key.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print(
            "OrientationInterpolator : set_fraction = {0}".format(set_fraction)
        )
        print(
            "OrientationInterpolator : key = {0}".format(key)
        )  # imprime no terminal
        print("OrientationInterpolator : keyValue = {0}".format(keyValue))

        # Abaixo está só um exemplo de como os dados podem ser calculados e transferidos
        value_changed = [0, 0, 1, 0]

        return value_changed

    # Para o futuro (Não para versão atual do projeto.)
    def vertex_shader(self, shader):
        """Para no futuro implementar um vertex shader."""

    def fragment_shader(self, shader):
        """Para no futuro implementar um fragment shader."""
