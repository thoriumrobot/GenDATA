/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 1997, 2021, Oracle and/or its affiliates. All rights reserved.
    @Positive
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
    @Positive
 *
    @Positive
 * This code is free software; you can redistribute it and/or modify it
    @Positive
 * under the terms of the GNU General Public License version 2 only, as
    @Positive
 * published by the Free Software Foundation.  Oracle designates this
    @Positive
 * particular file as subject to the "Classpath" exception as provided
    @Positive
 * by Oracle in the LICENSE file that accompanied this code.
    @Positive
 *
    @Positive
 * This code is distributed in the hope that it will be useful, but WITHOUT
    @Positive
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    @Positive
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    @Positive
 * version 2 for more details (a copy is included in the LICENSE file that
    @Positive
 * accompanied this code).
    @Positive
 *
    @Positive
 * You should have received a copy of the GNU General Public License version
    @Positive
 * 2 along with this work; if not, write to the Free Software Foundation,
    @Positive
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
    @Positive
 *
    @Positive
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
    @Positive
 * or visit www.oracle.com if you need additional information or have any
    @Positive
 * questions.
    @Positive
 */
    @Positive
package java.awt.geom;

    @Positive
import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
    @Positive
import org.checkerframework.checker.nullness.qual.NonNull;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import java.io.IOException;
    @Positive
import java.io.Serial;
    @Positive
import java.io.Serializable;

    @Positive
public abstract class Arc2D extends RectangularShape {

    @Positive
    public static final int OPEN;

    @Positive
    public static final int CHORD;

    @Positive
    public static final int PIE;

    @Positive
    public static class Float extends Arc2D implements Serializable {

    @Positive
        public float x;

    @Positive
        public float y;

    @Positive
        public float width;

    @Positive
        public float height;

    @Positive
        public float start;

    @Positive
        public float extent;

    @Positive
        public Float() {
    @Positive
        }

    @Positive
        public Float(int type) {
    @Positive
        }

    @Positive
        public Float(float x, float y, float w, float h, float start, float extent, int type) {
    @Positive
        }

    @Positive
        public Float(Rectangle2D ellipseBounds, float start, float extent, int type) {
    @Positive
        }

    @Positive
        public double getX();

    @Positive
        public double getY();

    @Positive
        public double getWidth();

    @Positive
        public double getHeight();

    @Positive
        public double getAngleStart();

    @Positive
        public double getAngleExtent();

    @Positive
        public boolean isEmpty();

    @Positive
        public void setArc(double x, double y, double w, double h, double angSt, double angExt, int closure);

    @Positive
        public void setAngleStart(double angSt);

    @Positive
        public void setAngleExtent(double angExt);

    @Positive
        protected Rectangle2D makeBounds(double x, double y, double w, double h);
    @Positive
    }

    @Positive
    public static class Double extends Arc2D implements Serializable {

    @Positive
        public double x;

    @Positive
        public double y;

    @Positive
        public double width;

    @Positive
        public double height;

    @Positive
        public double start;

    @Positive
        public double extent;

    @Positive
        public Double() {
    @Positive
        }

    @Positive
        public Double(int type) {
    @Positive
        }

    @Positive
        public Double(double x, double y, double w, double h, double start, double extent, int type) {
    @Positive
        }

    @Positive
        public Double(Rectangle2D ellipseBounds, double start, double extent, int type) {
    @Positive
        }

    @Positive
        public double getX();

    @Positive
        public double getY();

    @Positive
        public double getWidth();

    @Positive
        public double getHeight();

    @Positive
        public double getAngleStart();

    @Positive
        public double getAngleExtent();

    @Positive
        public boolean isEmpty();

    @Positive
        public void setArc(double x, double y, double w, double h, double angSt, double angExt, int closure);

    @Positive
        public void setAngleStart(double angSt);

    @Positive
        public void setAngleExtent(double angExt);

    @Positive
        protected Rectangle2D makeBounds(double x, double y, double w, double h);
    @Positive
    }

    @Positive
    protected Arc2D() {
    @Positive
    }

    @Positive
    protected Arc2D(int type) {
    @Positive
    }

    @Positive
    public abstract double getAngleStart();

    @Positive
    public abstract double getAngleExtent();

    @Positive
    public int getArcType();

    @Positive
    public Point2D getStartPoint();

    @Positive
    public Point2D getEndPoint();

    @Positive
    public abstract void setArc(double x, double y, double w, double h, double angSt, double angExt, int closure);

    @Positive
    public void setArc(Point2D loc, Dimension2D size, double angSt, double angExt, int closure);

    @Positive
    public void setArc(Rectangle2D rect, double angSt, double angExt, int closure);

    @Positive
    public void setArc(Arc2D a);

    @Positive
    public void setArcByCenter(double x, double y, double radius, double angSt, double angExt, int closure);

    @Positive
    public void setArcByTangent(Point2D p1, Point2D p2, Point2D p3, double radius);

    @Positive
    public abstract void setAngleStart(double angSt);

    @Positive
    public abstract void setAngleExtent(double angExt);

    @Positive
    public void setAngleStart(Point2D p);

    @Positive
    public void setAngles(double x1, double y1, double x2, double y2);

    @Positive
    public void setAngles(Point2D p1, Point2D p2);

    @Positive
    public void setArcType(int type);

    @Positive
    public void setFrame(double x, double y, double w, double h);

    @Positive
    public Rectangle2D getBounds2D();

    @Positive
    protected abstract Rectangle2D makeBounds(double x, double y, double w, double h);

    @Positive
    static double normalizeDegrees(double angle);

    @Positive
    public boolean containsAngle(double angle);

    @Positive
    public boolean contains(double x, double y);

    @Positive
    public boolean intersects(double x, double y, double w, double h);

    @Positive
    public boolean contains(double x, double y, double w, double h);

    @Positive
    public boolean contains(Rectangle2D r);

    @Positive
    public PathIterator getPathIterator(AffineTransform at);

    @Positive
    public int hashCode();

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object obj);
    @Positive
}
