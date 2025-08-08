use alumina::data::DataStream;
use alumina::graph::GraphDef;
use ndarray::ArrayD;

pub struct Validator {
	validation_stream: Option<Box<dyn DataStream>>,
	max_images: Option<usize>,
	epoch_size: usize,
}

impl Validator {
	pub fn new() -> Self {
		Self {
			validation_stream: None,
			max_images: None,
			epoch_size: 0,
		}
	}

	pub fn setup(
		&mut self,
		_graph: &GraphDef,
		_solver: &dyn alumina::opt::Opt,
		validation_stream: Box<dyn DataStream>,
		epoch_size: usize,
		max_images: Option<usize>,
		_is_prescaled: bool,
	) -> alumina::graph::Result<()> {
		self.validation_stream = Some(validation_stream);
		self.max_images = max_images;
		self.epoch_size = epoch_size;
		Ok(())
	}

	pub fn validate(&mut self, _params: &[ArrayD<f32>]) {
		println!("Validation skipped in refactored version");
	}
}
