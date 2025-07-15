def obj2vtk(input_obj):
    with open(input_obj, 'r') as obj_file:
        lines = obj_file.readlines()

    vertices, faces, transports = [], [], []

    for line in lines:
        tokens = line.split()
        if len(tokens) == 0:
            continue

        flag = tokens[0]

        if flag == 'v':
            vertices.append((float(tokens[1]), float(tokens[2]), float(tokens[3])))
            continue
        
        if flag == 'f':
            face = [int(vertex.split('/')[0]) for vertex in tokens[1:]]
            faces.append(face)
            continue

        if flag == 'vt':
            transports.append((float(tokens[1]), float(tokens[2]), float(tokens[3])))
            continue

    output_vtk = input_obj.split('.')[0] + '.vtk'
    with open(output_vtk, 'w') as vtk_file:
        vtk_file.write("# vtk DataFile Version 1.0\n")
        vtk_file.write("Mesh from obj\n")
        vtk_file.write("ASCII\n")
        vtk_file.write("DATASET UNSTRUCTURED_GRID\n")
        vtk_file.write("POINTS {} float\n".format(len(vertices)))

        for vertex in vertices:
            vtk_file.write("{:.6f} {:.6f} {:.6f}\n".format(vertex[0], vertex[1], vertex[2]))

        vtk_file.write("\nCELLS {} {}\n".format(len(faces), sum(len(face) + 1 for face in faces)))

        for face in faces:
            vtk_file.write("3 {} {} {}\n".format(face[0] - 1, face[1] - 1, face[2] - 1))

        vtk_file.write("\nCELL_TYPES {}\n".format(len(faces)))

        for _ in faces:
            vtk_file.write("5 ")

        vtk_file.write("\n\nPOINT_DATA {}\n".format(len(vertices)))

        vtk_file.write("\nVECTORS fields float\n")

        for transport in transports:
            vtk_file.write("{:.6f} {:.6f} {:.6f}\n".format(transport[0], transport[1], transport[2]))      

        vtk_file.write("\n")

def addVelocities_obj2vtk(input_obj, velocities):
    inputObj = input_obj.replace('/','.').split('.')
    output_obj = 'output/' + inputObj[1] + 'PT' + '.' + inputObj[-1]

    with open(input_obj, 'r') as obj_file:
        lines = obj_file.readlines()

    with open(output_obj, 'w') as new_obj_file:
        for line in lines:
            new_obj_file.write(line)

        new_obj_file.write('\n# Parallel Transport\n')
        for i, velocity in enumerate(velocities, start=1):
            new_obj_file.write("vt {:.6f} {:.6f} {:.6f}\n".format(velocity[0], velocity[1], velocity[2]))

    obj2vtk(output_obj)