import threading
import random
from tkinter import *
master = Tk()
canvas_width = 500
canvas_height = 700
input_start = [30, 30]
output_start = [480, 30]

w = Canvas(master,
		   width=canvas_width,
		   height=canvas_height)

w.pack()

def init(g):
	t = Worker()
	t.start()
	t.join()
	pass

def draw_status(gen, organisms , fitness, species, max_organisms, gnome):
	w.delete('all')
	reset()
	draw(gnome)
	w.create_rectangle(0,450,500,500,fill='#FFFFFF')
	t= "Gen: "+str(gen)+"   "+"Species: "+str(species)+"   "+"Fitness: "+ str(fitness)+"    "+str(organisms)+"/ "+str(max_organisms)
	w.create_text(250,460,text=t)
	w.create_rectangle(1,500,500,700,fill='#FFEEFF')
	cx = 1
	cy = 500
	cw = 50
	ch = 50
	for gene in gnome.genes:
		w.create_rectangle(cx, cy, cx + cw, cy+ch, fill="#FFFFFF")
		tex = str(gene.innovation_number)
		w.create_text(cw/2 + cx,ch/2 + cy,text=tex)
		tex = str(gene.input_node)+"->"+str(gene.output_node)
		w.create_text(cw/2 + cx,ch/2 + cy+10,text=tex)
		cx += cw
		if(cx+cw > canvas_width):
			cx = 0
			cy+=ch
	master.update()
	pass

cPoints = []
nonce = 0
def draw(g):
	
	cOutputs = []
	global cPoints
	# cPoints= []
	global nonce
	w.delete('all')
	if nonce == 0:
		max_y = 0
		x = input_start[0]
		y = input_start[1]
		for node in g.nodes:
			if node.ntype == 0:
				cPoints.append([x,y])
				circle(w,x,y,5)
				y += 15
				
		max_y = y
		x = output_start[0]
		y = output_start[1]
		for node in g.nodes:
			if node.ntype == 2:
				cPoints.append([x,y])
				circle(w, x, y, 5)
				y+=15

		x = 0
		y = 0
		for node in g.nodes:
			if node.ntype == 1:
				x = random.randrange(input_start[0]+50, output_start[0]-50)
				y = random.randrange(input_start[1]+50, max_y)
				cPoints.append([x, y])
				circle(w, x, y, 5)
		nonce = 1
	for point in cPoints:
		circle(w, point[0], point[1], 5)
	
	for gene in g.genes:
		c1 = cPoints[gene.input_node - 1]
		if (gene.output_node - 1) > len(cPoints):
			print(len(cPoints))
			print(gene.output_node)
		else:
			c2 = cPoints[gene.output_node -1]
		
			inp = g.nodes[gene.input_node - 1]
			h = abs(inp.output_value)*10000/32
			color = format(int(h), 'x')
			color += format(int(h), 'x')
			color += format(int(h), 'x')
			# print(color + str(inp.output_value))
			if gene.enabled == True and gene.weight != 0.0:
				w.create_line(c1[0], c1[1], c2[0], c2[1], fill="#"+color)
			
			# gene.weight
			# gene.enabled
		master.update()
	pass

def reset():
	global nonce 
	global cPoints
	cPoints = []
	nonce = 0

def circle(canvas, x, y, r):
   id = canvas.create_oval(x-r, y-r, x+r, y+r, fill='#FF0000')
   return id


# #w.create_rectangle(50, 20, 150, 80, fill="#476042")
# #w.create_line(0, y, canvas_width, y, fill="#476042")


