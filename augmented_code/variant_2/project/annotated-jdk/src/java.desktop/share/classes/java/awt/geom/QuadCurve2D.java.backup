/*
 * Copyright (c) 1997, 2021, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.  Oracle designates this
 * particular file as subject to the "Classpath" exception as provided
 * by Oracle in the LICENSE file that accompanied this code.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
 * or visit www.oracle.com if you need additional information or have any
 * questions.
 */
package java.awt.geom;

import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.awt.Rectangle;
import java.awt.Shape;
import java.io.Serial;
import java.io.Serializable;

@AnnotatedFor({ "interning" })
@UsesObjectEquals
public abstract class QuadCurve2D implements Shape, Cloneable {

    public static class Float extends QuadCurve2D implements Serializable {

        public float x1;

        public float y1;

        public float ctrlx;

        public float ctrly;

        public float x2;

        public float y2;

        public Float() {
        }

        public Float(float x1, float y1, float ctrlx, float ctrly, float x2, float y2) {
        }

        public double getX1();

        public double getY1();

        public Point2D getP1();

        public double getCtrlX();

        public double getCtrlY();

        public Point2D getCtrlPt();

        public double getX2();

        public double getY2();

        public Point2D getP2();

        public void setCurve(double x1, double y1, double ctrlx, double ctrly, double x2, double y2);

        public void setCurve(float x1, float y1, float ctrlx, float ctrly, float x2, float y2);

        public Rectangle2D getBounds2D();
    }

    public static class Double extends QuadCurve2D implements Serializable {

        public double x1;

        public double y1;

        public double ctrlx;

        public double ctrly;

        public double x2;

        public double y2;

        public Double() {
        }

        public Double(double x1, double y1, double ctrlx, double ctrly, double x2, double y2) {
        }

        public double getX1();

        public double getY1();

        public Point2D getP1();

        public double getCtrlX();

        public double getCtrlY();

        public Point2D getCtrlPt();

        public double getX2();

        public double getY2();

        public Point2D getP2();

        public void setCurve(double x1, double y1, double ctrlx, double ctrly, double x2, double y2);

        public Rectangle2D getBounds2D();
    }

    protected QuadCurve2D() {
    }

    public abstract double getX1();

    public abstract double getY1();

    public abstract Point2D getP1();

    public abstract double getCtrlX();

    public abstract double getCtrlY();

    public abstract Point2D getCtrlPt();

    public abstract double getX2();

    public abstract double getY2();

    public abstract Point2D getP2();

    public abstract void setCurve(double x1, double y1, double ctrlx, double ctrly, double x2, double y2);

    public void setCurve(double[] coords, int offset);

    public void setCurve(Point2D p1, Point2D cp, Point2D p2);

    public void setCurve(Point2D[] pts, int offset);

    public void setCurve(QuadCurve2D c);

    public static double getFlatnessSq(double x1, double y1, double ctrlx, double ctrly, double x2, double y2);

    public static double getFlatness(double x1, double y1, double ctrlx, double ctrly, double x2, double y2);

    public static double getFlatnessSq(double[] coords, int offset);

    public static double getFlatness(double[] coords, int offset);

    public double getFlatnessSq();

    public double getFlatness();

    public void subdivide(QuadCurve2D left, QuadCurve2D right);

    public static void subdivide(QuadCurve2D src, QuadCurve2D left, QuadCurve2D right);

    public static void subdivide(double[] src, int srcoff, double[] left, int leftoff, double[] right, int rightoff);

    public static int solveQuadratic(double[] eqn);

    public static int solveQuadratic(double[] eqn, double[] res);

    public boolean contains(double x, double y);

    public boolean contains(Point2D p);

    public boolean intersects(double x, double y, double w, double h);

    public boolean intersects(Rectangle2D r);

    public boolean contains(double x, double y, double w, double h);

    public boolean contains(Rectangle2D r);

    public Rectangle getBounds();

    public PathIterator getPathIterator(AffineTransform at);

    public PathIterator getPathIterator(AffineTransform at, double flatness);

    public Object clone();
}
