import numpy as np

class TemplateGenerator():

	def __init__(self, shape_model=None, cell_size=None):
		""" Instantiates TempalteGenerator. Creates default shape model if none provided"""

		if not shape_model:

			# Number of pixels in each cell
			self.cell_size = 6

			# Make dummy data shape model
			shape_model = np.zeros([20,10])
			shape_model[2:4,4:6] = 1
			shape_model[4:11,2:8] = 2
			shape_model[11:18,3:7] = 3
			self.shape_model = shape_model

		else:

			self.cell_size = cell_size
			self.shape_model = shape_model


	def generate_sizes(self, w_max=4, h_max=3):
		""" Generates set of possible template sizes """

		# Define width and height constraints in terms of cells
		w_vals = range(1, w_max + 1)
		h_vals = range(1, h_max + 1)

		# Generate size pool for template sizes
		sizes = [(w,h) for w in w_vals for h in h_vals]
		self.sizes = sizes[1:]

	def generate_templates(self):
		""" Generates templates by convolving windows defined by sizes over the shape model """

		templates = []
		cell_size = self.cell_size

		# Slide each size template over the entire shape model and generate templates
		for size in self.sizes:
			w = size[0]
			h = size[1]

			# Slide template with dimenions specified by size across the entire shape model
			for y in xrange(self.shape_model.shape[0] - h + 1):
				for x in xrange(self.shape_model.shape[1] - w + 1):

					mat_temp = np.copy(self.shape_model[y:y+h, x:x+w])
					unique = np.unique(mat_temp)

					# Check to make sure template holds some shape model information
					if len(unique) > 1:

						# Binary template: set values to 1 and 0 and add template
						if len(unique) == 2:
							mat_temp[mat_temp == unique[0]] = 1
							mat_temp[mat_temp == unique[1]] = 0
							templates.append((w*cell_size,y*cell_size,size,mat_temp))

						# Ternary template: set values to -1, 0, 1 -- add template -- repeat with all permutations
						else:
							mat_temp[mat_temp == unique[0]] = -1
							mat_temp[mat_temp == unique[1]] = 0
							mat_temp[mat_temp == unique[2]] = 1
							templates.append((w*cell_size,y*cell_size,size,mat_temp))

							mat_temp[mat_temp == -1] = 0
							mat_temp[mat_temp == 0] = 1
							mat_temp[mat_temp == 1] = -1
							templates.append((w*cell_size,y*cell_size,size,mat_temp))

							mat_temp[mat_temp == -1] = 1
							mat_temp[mat_temp == 0] = -1
							mat_temp[mat_temp == 1] = 0
							templates.append((w*cell_size,y*cell_size,size,mat_temp))


		self.templates = templates
		return self.templates
