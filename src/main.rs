use nalgebra::Vector3;

// import types into main
mod types;
use types::Body;

const G: f64 = 6.6743e-11;
const dt : f64 = 60.0 * 60.0; // 1h


fn Initialize_bodies() -> Vec<Body>{
    
    vec![
        Body::new(2.0, Vector3::new(5.0, 0.0, 0.0), Vector3::new(0.0, 0.0, 0.0)),
        Body::new(1.0, Vector3::new(1.0, 0.0, 0.0), Vector3::new(0.0, 1.0, 0.0)),
        Body::new(0.5, Vector3::new(1.5, 0.0, 0.0), Vector3::new(0.0, 0.8, 0.0)),
    ] // Implicit return of the Vec<Body>
}

fn Reset_forces(bodies: &mut Vec<Body>){
    for body in bodies{
        body.force = Vector3::new(0.0,0.0,0.0);
    }
    println!("reseted forces!");
}

fn Calculate_forces(bodies: &mut Vec<Body>){
    let num_bodies = bodies.len();

    println!("body len {}",num_bodies);

    for i in 0..num_bodies{
        for j in (i + 1)..num_bodies {

            let r_vec : Vector3<f64>= bodies[i].position - bodies[j].position;
            let r_scalar : f64 = r_vec.norm();

            if r_scalar < 1e-6 { continue; }

            let magnitude_factor : f64 = G * bodies[i].mass * bodies[j].mass / r_scalar.powi(3);
            println!("magnitude_factor {}",magnitude_factor);
            let force_vec : Vector3<f64> = magnitude_factor * r_vec;
            println!("force {}",force_vec);

            bodies[i].force += force_vec;
            bodies[j].force -= force_vec;
        }
    }
}

fn Calculate_pos(bodies: &mut Vec<Body>){
    for mut body in bodies{
        let accel = body.force / body.mass;
        body.velocity +=accel * dt;
        body.position += body.velocity * dt
    }
}



fn main() {
    println!("Simulation Constants Initialized:");
    println!("G = {} m³⋅kg⁻¹⋅s⁻²", G);
    println!("Δt = {} seconds", dt);

    let mut bodies : Vec<Body> = Initialize_bodies();

    Reset_forces(&mut bodies);
    Calculate_forces(&mut bodies);
    Calculate_pos(&mut bodies);
    
}