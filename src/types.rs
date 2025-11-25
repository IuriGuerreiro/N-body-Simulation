use nalgebra::Vector3;

pub struct Body {
    pub mass : f64,
    pub velocity : Vector3<f64>,
    pub position : Vector3<f64>,
    pub force : Vector3<f64>,
}

impl Body {
    // A simplified constructor only requiring mass, position, and velocity
    pub fn new(mass: f64, position: Vector3<f64>, velocity: Vector3<f64>) -> Self {
        Body {
            mass,
            position,
            velocity,
            // Initializes force to zero
            force: Vector3::new(0.0, 0.0, 0.0), 
        }
    }
}