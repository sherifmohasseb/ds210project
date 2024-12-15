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

fn gauss(a: Vec<Vec<f64>>, b: Vec<f64>) -> Vec<f64> {
    let n = a.len();
    let mut aug = vec![vec![0.0; n+1]; n];
    for i in 0..n {
        for j in 0..n {
            aug[i][j] = a[i][j];
        }
        aug[i][n] = b[i];
    }

    for i in 0..n {
        let mut pv = i;
        for r in i+1..n {
            if aug[r][i].abs() > aug[pv][i].abs() {
                pv = r;
            }
        }
        if aug[pv][i].abs() < 1e-12 {
            return vec![];
        }
        if pv != i {
            aug.swap(i, pv);
        }
        let pval = aug[i][i];
        for c in i..n+1 {
            aug[i][c] /= pval;
        }
        for r in i+1..n {
            let f = aug[r][i];
            for c in i..n+1 {
                aug[r][c] -= f * aug[i][c];
            }
        }
    }

    for i in (0..n).rev() {
        for r in 0..i {
            let f = aug[r][i];
            aug[r][n] -= f * aug[i][n];
            aug[r][i] = 0.0;
        }
    }

    (0..n).map(|i| aug[i][n]).collect()
}

fn fit_linreg(data: &[(f64, f64, f64, f64)]) -> (f64, f64, f64, f64) {
    let n = data.len();
    if n < 4 {
        eprint!("not enough data\n");
        return (0.0,0.0,0.0,0.0);
    }

    let mut X = Array2::<f64>::zeros((n,4));
    let mut Y = Array1::<f64>::zeros(n);

    for (i, &(km, yr, eng, pr)) in data.iter().enumerate() {
        X[[i,0]] = 1.0;
        X[[i,1]] = km;
        X[[i,2]] = yr;
        X[[i,3]] = eng;
        Y[i] = pr;
    }

    let xtx = X.t().dot(&X);
    let xty = X.t().dot(&Y);

    let xtx_vec: Vec<Vec<f64>> = (0..4).map(|i| {
        (0..4).map(|j| xtx[[i,j]]).collect()
    }).collect();

    let xty_vec: Vec<f64> = (0..4).map(|i| xty[i]).collect();

    let b = gauss(xtx_vec, xty_vec);
    if b.len() == 4 {
        (b[0], b[1], b[2], b[3])
    } else {
        eprint!("cannot solve\n");
        (0.0,0.0,0.0,0.0)
    }
}
