/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Copyright * Positive (c) 1997, 2021, Oracle and/or its affiliates. All rights reserved.
    @DO * Positive NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
    @Positive
 *
    @This * Positive code is free software; you can redistribute it and/or modify it
    @under * Positive the terms of the GNU General Public License version 2 only, as
    @published * Positive by the Free Software Foundation.  Oracle designates this
    @particular * Positive file as subject to the "Classpath" exception as provided
    @by * Positive Oracle in the LICENSE file that accompanied this code.
    @Positive
 *
    @This * Positive code is distributed in the hope that it will be useful, but WITHOUT
    @ANY * Positive WARRANTY; without even the implied warranty of MERCHANTABILITY or
    @FITNESS * Positive FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    @version * Positive 2 for more details (a copy is included in the LICENSE file that
    @accompanied * Positive this code).
    @Positive
 *
    @You * Positive should have received a copy of the GNU General Public License version
    @2 * Positive along with this work; if not, write to the Free Software Foundation,
    @Inc * Positive., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
    @Positive
 *
    @Please * Positive contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
    @or * Positive visit www.oracle.com if you need additional information or have any
    @questions * Positive.
    @Positive
 */
    @Positive
package java.awt.geom;

    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.awt.Rectangle;
    @Positive
import java.awt.Shape;
    @Positive
import java.io.Serial;
    @Positive
import java.io.Serializable;
    @Positive
import java.util.Arrays;
    @Positive
import sun.awt.geom.Curve;
    @Positive
import static java.lang.Math.abs;
    @Positive
import static java.lang.Math.max;
    @Positive
import static java.lang.Math.ulp;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
public abstract class CubicCurve2D implements Shape, Cloneable {

    @Positive
    public static class Float extends CubicCurve2D implements Serializable {

    @Positive
        public float x1;

    @Positive
        public float y1;

    @Positive
        public float ctrlx1;

    @Positive
        public float ctrly1;

    @Positive
        public float ctrlx2;

    @Positive
        public float ctrly2;

    @Positive
        public float x2;

    @Positive
        public float y2;

    @Positive
        public Float() {
    @Positive
        }

    @Positive
        public Float(float x1, float y1, float ctrlx1, float ctrly1, float ctrlx2, float ctrly2, float x2, float y2) {
    @Positive
        }

    @Positive
        public double getX1();

    @Positive
        public double getY1();

    @Positive
        public Point2D getP1();

    @Positive
        public double getCtrlX1();

    @Positive
        public double getCtrlY1();

    @Positive
        public Point2D getCtrlP1();

    @Positive
        public double getCtrlX2();

    @Positive
        public double getCtrlY2();

    @Positive
        public Point2D getCtrlP2();

    @Positive
        public double getX2();

    @Positive
        public double getY2();

    @Positive
        public Point2D getP2();

    @Positive
        public void setCurve(double x1, double y1, double ctrlx1, double ctrly1, double ctrlx2, double ctrly2, double x2, double y2);

    @Positive
        public void setCurve(float x1, float y1, float ctrlx1, float ctrly1, float ctrlx2, float ctrly2, float x2, float y2);

    @Positive
        public Rectangle2D getBounds2D();
    @Positive
    }

    @Positive
    public static class Double extends CubicCurve2D implements Serializable {

    @Positive
        public double x1;

    @Positive
        public double y1;

    @Positive
        public double ctrlx1;

    @Positive
        public double ctrly1;

    @Positive
        public double ctrlx2;

    @Positive
        public double ctrly2;

    @Positive
        public double x2;

    @Positive
        public double y2;

    @Positive
        public Double() {
    @Positive
        }

    @Positive
        public Double(double x1, double y1, double ctrlx1, double ctrly1, double ctrlx2, double ctrly2, double x2, double y2) {
    @Positive
        }

    @Positive
        public double getX1();

    @Positive
        public double getY1();

    @Positive
        public Point2D getP1();

    @Positive
        public double getCtrlX1();

    @Positive
        public double getCtrlY1();

    @Positive
        public Point2D getCtrlP1();

    @Positive
        public double getCtrlX2();

    @Positive
        public double getCtrlY2();

    @Positive
        public Point2D getCtrlP2();

    @Positive
        public double getX2();

    @Positive
        public double getY2();

    @Positive
        public Point2D getP2();

    @Positive
        public void setCurve(double x1, double y1, double ctrlx1, double ctrly1, double ctrlx2, double ctrly2, double x2, double y2);

    @Positive
        public Rectangle2D getBounds2D();
    @Positive
    }

    @Positive
    protected CubicCurve2D() {
    @Positive
    }

    @Positive
    public abstract double getX1();

    @Positive
    public abstract double getY1();

    @Positive
    public abstract Point2D getP1();

    @Positive
    public abstract double getCtrlX1();

    @Positive
    public abstract double getCtrlY1();

    @Positive
    public abstract Point2D getCtrlP1();

    @Positive
    public abstract double getCtrlX2();

    @Positive
    public abstract double getCtrlY2();

    @Positive
    public abstract Point2D getCtrlP2();

    @Positive
    public abstract double getX2();

    @Positive
    public abstract double getY2();

    @Positive
    public abstract Point2D getP2();

    @Positive
    public abstract void setCurve(double x1, double y1, double ctrlx1, double ctrly1, double ctrlx2, double ctrly2, double x2, double y2);

    @Positive
    public void setCurve(double[] coords, int offset);

    @Positive
    public void setCurve(Point2D p1, Point2D cp1, Point2D cp2, Point2D p2);

    @Positive
    public void setCurve(Point2D[] pts, int offset);

    @Positive
    public void setCurve(CubicCurve2D c);

    @Positive
    public static double getFlatnessSq(double x1, double y1, double ctrlx1, double ctrly1, double ctrlx2, double ctrly2, double x2, double y2);

    @Positive
    public static double getFlatness(double x1, double y1, double ctrlx1, double ctrly1, double ctrlx2, double ctrly2, double x2, double y2);

    @Positive
    public static double getFlatnessSq(double[] coords, int offset);

    @Positive
    public static double getFlatness(double[] coords, int offset);

    @Positive
    public double getFlatnessSq();

    @Positive
    public double getFlatness();

    @Positive
    public void subdivide(CubicCurve2D left, CubicCurve2D right);

    @Positive
    public static void subdivide(CubicCurve2D src, CubicCurve2D left, CubicCurve2D right);

    @Positive
    public static void subdivide(double[] src, int srcoff, double[] left, int leftoff, double[] right, int rightoff);

    @Positive
    public static int solveCubic(double[] eqn);

    @Positive
    public static int solveCubic(double[] eqn, double[] res);

    @Positive
    public boolean contains(double x, double y);

    @Positive
    public boolean contains(Point2D p);

    @Positive
    public boolean intersects(double x, double y, double w, double h);

    @Positive
    public boolean intersects(Rectangle2D r);

    @Positive
    public boolean contains(double x, double y, double w, double h);

    @Positive
    public boolean contains(Rectangle2D r);

    @Positive
    public Rectangle getBounds();

    @Positive
    public PathIterator getPathIterator(AffineTransform at);

    @Positive
    public PathIterator getPathIterator(AffineTransform at, double flatness);

    @Positive
    public Object clone();
    @Positive
}
