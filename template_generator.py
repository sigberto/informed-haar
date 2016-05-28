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
							idx1 = mat_temp == unique[0]
							idx2 = mat_temp == unique[1]
							
							mat_temp[idx1] = 1
							mat_temp[idx2] = 0
							templates.append((x,y,size,mat_temp))

						# Ternary template: set values to -1, 0, 1 -- add template -- repeat with all permutations
						else:
							# Get unique value indices
							idx1 = mat_temp == unique[0]
							idx2 = mat_temp == unique[1]
							idx3 = mat_temp == unique[2]
							
							mat_temp[idx1] = -1
							mat_temp[idx2] = 0
							mat_temp[idx3] = 1
							templates.append((x,y,size,mat_temp))
							
							mat_temp[idx1] = 1
							mat_temp[idx2] = -1
							mat_temp[idx3] = 0
							templates.append((x,y,size,mat_temp))
							
							mat_temp[idx1] = 0
							mat_temp[idx2] = 1
							mat_temp[idx3] = -1
							templates.append((x,y,size,mat_temp))


		self.templates = np.asarray(templates, dtype=object)
		self.remove_duplicates()
		self.shift_templates()
		self.normalize_templates()
		return self.templates
		
	def remove_duplicates(self):
		""" Removes all duplicate templates """
		
		to_remove = []
		
		# Compare every template against each other
		for idx, t1 in enumerate(self.templates):
			for idx2, t2 in enumerate(self.templates[idx+1:]):
				
				#If templates at the same x,y coordinate
				if t1[0] == t2[0] and t1[1] == t2[1]:
					_, _, size1, W1 = t1
					_, _, size2, W2 = t2
					w1, h1 = size1
					w2, h2 = size2
					wmax = max([w1,w2])
					hmax = max([h1,h2])

					#Expand matrices
					W1p = np.zeros([hmax, wmax])
					W2p = np.zeros([hmax, wmax])
					W1p[:h1,:w1] = W1
					W2p[:h2,:w2] = W2
					
					
					#If matrices subtracted from each other == 0, remove one
					if np.sum(np.abs(W1p - W2p)) == 0:
						to_remove.append(idx)
						break
				
		# Get indices for subset of templates     
		indices = [x for x in range(len(self.templates)) if x not in to_remove]
		self.templates = self.templates[indices]
		
	def shift_templates(self):
		
		new_templates = []
		
		# Iterate through each template and add new template/shift up, down, left, right one cell if possible.
		for t in self.templates:
			x, y, size, W = t
						
			if y < self.shape_model.shape[0] - 1:
				new_templates.append((x,y+1,size,W))
				
			if y > 0:
				new_templates.append((x,y-1,size,W))
				
			if x < self.shape_model.shape[1] -1:
				new_templates.append((x+1,y,size,W))
				
			if x > 0:
				new_templates.append((x-1,y,size,W))
				
		new_templates = np.asarray(new_templates, dtype=object)
		
		self.templates = np.concatenate((self.templates,new_templates),axis=0)
		
	def normalize_templates(self):
		
		for idx, t in enumerate(self.templates):
			
			x,y,size,W = t
			
			W1 = np.copy(W)
			W2 = np.copy(W)
			
			W1[W1 != 1] = 0
			W2[W2 != -1] = 0

			s1 = np.sum(W1)
			s2 = np.sum(-W2)
			
			if s2:
				self.templates[idx] = (x,y,size,np.copy(W1/s1 + W2/s2))
			else:
				self.templates[idx] = (x,y,size,np.copy(W1/s1))


