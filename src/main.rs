use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;
use ndarray::{Array2, Array1};

fn load_csv(fp: &str) -> Vec<(f64, f64, f64, f64)> {
    let mut data = Vec::new();
    let mut skip = 0;

    if !Path::new(fp).exists() {
        eprint!("err: '{}' not found\n", fp);
        return data;
    }

    println!("loading '{}'", fp);
    if let Ok(lines) = rl(fp) {
        for (i, l) in lines.enumerate() {
            if let Ok(rec) = l {
                if i == 0 {
                    continue;
                }
                let flds: Vec<&str> = rec.split(',').map(|x| x.trim()).collect();
                if flds.len() < 12 {
                    skip += 1;
                    continue;
                }

                let Year_raw = flds[5];
                let Engine_raw = flds[7];
                let Kilometers_raw = flds[8];
                let Price_raw = flds[11];

                if is_num(Year_raw) && is_num(Engine_raw) && is_num(Kilometers_raw) && is_num(Price_raw) {
                    let Year = Year_raw.parse::<f64>().unwrap();
                    let Engine = Engine_raw.parse::<f64>().unwrap();
                    let Kilometers = Kilometers_raw.parse::<f64>().unwrap();
                    let Price = Price_raw.parse::<f64>().unwrap();
                    data.push((Kilometers, Year, Engine, Price));
                } else {
                    skip += 1;
                }
            } else {
                skip += 1;
            }
        }
    } else {
        eprint!("err: cannot open '{}'\n", fp);
    }

    if !data.is_empty() {
        println!("rows: {}", data.len());
    } else {
        println!("no data");
    }
    println!("skipped: {}", skip);

    data
}

fn is_num(v: &str) -> bool {
    v.parse::<f64>().is_ok()
}

fn rl<P>(file: P) -> io::Result<io::Lines<io::BufReader<File>>>
where P: AsRef<Path> {
    let f = File::open(file)?;
    Ok(io::BufReader::new(f).lines())
}

fn avg(d: &[f64]) -> f64 {
    if d.is_empty() { 0.0 } else { d.iter().sum::<f64>() / d.len() as f64 }
}

fn sd(d: &[f64]) -> f64 {
    if d.len() < 2 {
        0.0
    } else {
        let m = avg(d);
        let var = d.iter().map(|x| (x - m).powi(2)).sum::<f64>() / d.len() as f64;
        var.sqrt()
    }
}

fn correlation(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.is_empty() {
        return 0.0;
    }
    let mx = avg(x);
    let my = avg(y);
    let num: f64 = x.iter().zip(y.iter()).map(|(xi, yi)| (xi - mx)*(yi - my)).sum();
    let dx = x.iter().map(|xi| (xi - mx).powi(2)).sum::<f64>().sqrt();
    let dy = y.iter().map(|yi| (yi - my).powi(2)).sum::<f64>().sqrt();
    if dx == 0.0 || dy == 0.0 {0.0} else { num / (dx*dy) }
}
