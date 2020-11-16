//! Computational geometry functions, for example finding convex hulls.

use crate::point::{distance, Line, Point, Rotation};
use num::{cast, Num, NumCast};
use std::cmp::{Ord, Ordering};
use std::f64::{self, consts::PI};

/// Returns the length of the arc constructed with the provided points in
/// incremental order. When the `closed` param is set to `true`, the distance
/// between the last and the first point is included in the total length.
pub fn arc_length<T>(arc: &[Point<T>], closed: bool) -> f64
where
    T: Num + NumCast + Copy + PartialEq + Eq,
{
    let mut length = arc.windows(2).map(|pts| distance(pts[0], pts[1])).sum();

    if arc.len() > 2 && closed {
        length += distance(arc[0], arc[arc.len() - 1]);
    }

    length
}

/// Fits the polygon curve to a similar curve with fewer points.
/// The input parameters include an ordered array of points and an distance
/// dimension `epsilon` > 0. Based on the [Douglas–Peucker algorithm].
///
/// [Douglas–Peucker algorithm]: https://en.wikipedia.org/wiki/Ramer-Douglas-Peucker_algorithm
pub fn approx_poly_dp<T>(curve: &[Point<T>], epsilon: f64, closed: bool) -> Vec<Point<T>>
where
    T: Num + NumCast + Copy + PartialEq + Eq,
{
    if epsilon <= 0.0 {
        panic!("epsilon must be greater than 0.0");
    }
    // Find the point with the maximum distance
    let mut dmax = 0.0;
    let mut index = 0;
    let end = curve.len() - 1;
    let line = Line::from_points(curve[0].to_f64(), curve[end].to_f64());
    for (i, point) in curve.iter().enumerate().skip(1) {
        let d = line.distance_from_point(point.to_f64());
        if d > dmax {
            index = i;
            dmax = d;
        }
    }

    let mut res = Vec::new();

    // If max distance is greater than epsilon, recursively simplify
    if dmax > epsilon {
        // Recursive call
        let mut partial1 = approx_poly_dp(&curve[0..=index], epsilon, false);
        let mut partial2 = approx_poly_dp(&curve[index..=end], epsilon, false);

        // Build the result list
        partial1.pop();
        res.append(&mut partial1);
        res.append(&mut partial2);
    } else {
        res.push(curve[0]);
        res.push(curve[end]);
    }

    if closed {
        res.pop();
    }

    res
}

/// Finds the minimal area rectangle that covers all of the points in the input
/// contour in the following order -> (TL, TR, BR, BL).
pub fn min_area_rect<T>(contour: &[Point<T>]) -> [Point<T>; 4]
where
    T: Num + NumCast + Copy + PartialEq + Eq + Ord,
{
    let hull = convex_hull(&contour);
    match hull.len() {
        0 => panic!("no points are defined"),
        1 => [hull[0]; 4],
        2 => [hull[0], hull[1], hull[1], hull[0]],
        _ => rotating_calipers(&hull),
    }
}

/// The implementation of the [rotating calipers] used for determining the
/// bounding rectangle with the smallest area.
///
/// [rotating calipers]: https://en.wikipedia.org/wiki/Rotating_calipers
fn rotating_calipers<T>(points: &[Point<T>]) -> [Point<T>; 4]
where
    T: Num + NumCast + Copy + PartialEq + Eq,
{
    let mut edge_angles: Vec<f64> = points
        .windows(2)
        .map(|e| {
            let edge = e[1].to_f64() - e[0].to_f64();
            ((edge.y.atan2(edge.x) + PI) % (PI / 2.)).abs()
        })
        .collect();

    edge_angles.dedup();

    let mut min_area = f64::MAX;
    let mut res = vec![Point::new(0.0, 0.0); 4];
    for angle in edge_angles {
        let rotation = Rotation::new(angle);
        let rotated_points: Vec<Point<f64>> =
            points.iter().map(|p| p.to_f64().rotate(rotation)).collect();

        let (min_x, max_x, min_y, max_y) =
            rotated_points
                .iter()
                .fold((f64::MAX, f64::MIN, f64::MAX, f64::MIN), |acc, p| {
                    (
                        acc.0.min(p.x),
                        acc.1.max(p.x),
                        acc.2.min(p.y),
                        acc.3.max(p.y),
                    )
                });

        let area = (max_x - min_x) * (max_y - min_y);
        if area < min_area {
            min_area = area;
            res[0] = Point::new(max_x, min_y).invert_rotation(rotation);
            res[1] = Point::new(min_x, min_y).invert_rotation(rotation);
            res[2] = Point::new(min_x, max_y).invert_rotation(rotation);
            res[3] = Point::new(max_x, max_y).invert_rotation(rotation);
        }
    }

    res.sort_by(|a, b| {
        if a.x < b.x {
            Ordering::Less
        } else if a.x > b.x {
            Ordering::Greater
        } else {
            Ordering::Equal
        }
    });

    let i1 = if res[1].y > res[0].y { 0 } else { 1 };
    let i2 = if res[3].y > res[2].y { 2 } else { 3 };
    let i3 = if res[3].y > res[2].y { 3 } else { 2 };
    let i4 = if res[1].y > res[0].y { 1 } else { 0 };

    [
        Point::new(
            cast(res[i1].x.floor()).unwrap(),
            cast(res[i1].y.floor()).unwrap(),
        ),
        Point::new(
            cast(res[i2].x.ceil()).unwrap(),
            cast(res[i2].y.floor()).unwrap(),
        ),
        Point::new(
            cast(res[i3].x.ceil()).unwrap(),
            cast(res[i3].y.ceil()).unwrap(),
        ),
        Point::new(
            cast(res[i4].x.floor()).unwrap(),
            cast(res[i4].y.ceil()).unwrap(),
        ),
    ]
}

/// Finds the convex hull of a set of points, using the [Graham scan algorithm].
///
/// [Graham scan algorithm]: https://en.wikipedia.org/wiki/Graham_scan
pub fn convex_hull<T>(points_slice: &[Point<T>]) -> Vec<Point<T>>
where
    T: Num + NumCast + Copy + PartialEq + Eq + Ord,
{
    if points_slice.is_empty() {
        return Vec::new();
    }
    let mut points: Vec<Point<T>> = points_slice.to_vec();
    let mut start_point_pos = 0;
    let mut start_point = points[0];
    for (i, &point) in points.iter().enumerate().skip(1) {
        if point.y < start_point.y || point.y == start_point.y && point.x < start_point.x {
            start_point_pos = i;
            start_point = point;
        }
    }
    points.swap(0, start_point_pos);
    points.remove(0);
    points.sort_by(
        |a, b| match orientation(start_point.to_i32(), a.to_i32(), b.to_i32()) {
            Orientation::Collinear => {
                if distance(start_point, *a) < distance(start_point, *b) {
                    Ordering::Less
                } else {
                    Ordering::Greater
                }
            }
            Orientation::Clockwise => Ordering::Greater,
            Orientation::CounterClockwise => Ordering::Less,
        },
    );

    let mut iter = points.iter().peekable();
    let mut remaining_points = Vec::with_capacity(points.len());
    while let Some(mut p) = iter.next() {
        while iter.peek().is_some()
            && orientation(
                start_point.to_i32(),
                p.to_i32(),
                iter.peek().unwrap().to_i32(),
            ) == Orientation::Collinear
        {
            p = iter.next().unwrap();
        }
        remaining_points.push(p);
    }

    let mut stack: Vec<Point<T>> = vec![Point::new(
        cast(start_point.x).unwrap(),
        cast(start_point.y).unwrap(),
    )];

    for p in points {
        while stack.len() > 1
            && orientation(
                stack[stack.len() - 2].to_i32(),
                stack[stack.len() - 1].to_i32(),
                p.to_i32(),
            ) != Orientation::CounterClockwise
        {
            stack.pop();
        }
        stack.push(p);
    }
    stack
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum Orientation {
    Collinear,
    Clockwise,
    CounterClockwise,
}

fn orientation(p: Point<i32>, q: Point<i32>, r: Point<i32>) -> Orientation {
    let val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y);
    match val.cmp(&0) {
        Ordering::Equal => Orientation::Collinear,
        Ordering::Greater => Orientation::Clockwise,
        Ordering::Less => Orientation::CounterClockwise,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::point::Point;

    #[test]
    fn convex_hull_points() {
        let star = vec![
            Point::new(100, 20),
            Point::new(90, 35),
            Point::new(60, 25),
            Point::new(90, 40),
            Point::new(80, 55),
            Point::new(101, 50),
            Point::new(130, 60),
            Point::new(115, 45),
            Point::new(140, 30),
            Point::new(120, 35),
        ];
        let points = convex_hull(&star);
        assert_eq!(
            points,
            [
                Point::new(100, 20),
                Point::new(140, 30),
                Point::new(130, 60),
                Point::new(80, 55),
                Point::new(60, 25)
            ]
        );
    }

    #[test]
    fn convex_hull_points_empty_vec() {
        let points = convex_hull::<i32>(&vec![]);
        assert_eq!(points, []);
    }

    #[test]
    fn convex_hull_points_with_negative_values() {
        let star = vec![
            Point::new(100, -20),
            Point::new(90, 5),
            Point::new(60, -15),
            Point::new(90, 0),
            Point::new(80, 15),
            Point::new(101, 10),
            Point::new(130, 20),
            Point::new(115, 5),
            Point::new(140, -10),
            Point::new(120, -5),
        ];
        let points = convex_hull(&star);
        assert_eq!(
            points,
            [
                Point::new(100, -20),
                Point::new(140, -10),
                Point::new(130, 20),
                Point::new(80, 15),
                Point::new(60, -15)
            ]
        );
    }

    #[test]
    fn min_area_test() {
        assert_eq!(
            min_area_rect(&[
                Point::new(100, 20),
                Point::new(140, 30),
                Point::new(130, 60),
                Point::new(80, 55),
                Point::new(60, 25)
            ]),
            [
                Point::new(60, 16),
                Point::new(141, 24),
                Point::new(137, 61),
                Point::new(57, 53)
            ]
        )
    }
}
