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

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import java.io.IOException;
import java.io.Serial;
import java.io.Serializable;

public abstract class Arc2D extends RectangularShape {

    public static final int OPEN;

    public static final int CHORD;

    public static final int PIE;

    public static class Float extends Arc2D implements Serializable {

        public float x;

        public float y;

        public float width;

        public float height;

        public float start;

        public float extent;

        public Float() {
        }

        public Float(int type) {
        }

        public Float(float x, float y, float w, float h, float start, float extent, int type) {
        }

        public Float(Rectangle2D ellipseBounds, float start, float extent, int type) {
        }

        public double getX();

        public double getY();

        public double getWidth();

        public double getHeight();

        public double getAngleStart();

        public double getAngleExtent();

        public boolean isEmpty();

        public void setArc(double x, double y, double w, double h, double angSt, double angExt, int closure);

        public void setAngleStart(double angSt);

        public void setAngleExtent(double angExt);

        protected Rectangle2D makeBounds(double x, double y, double w, double h);
    }

    public static class Double extends Arc2D implements Serializable {

        public double x;

        public double y;

        public double width;

        public double height;

        public double start;

        public double extent;

        public Double() {
        }

        public Double(int type) {
        }

        public Double(double x, double y, double w, double h, double start, double extent, int type) {
        }

        public Double(Rectangle2D ellipseBounds, double start, double extent, int type) {
        }

        public double getX();

        public double getY();

        public double getWidth();

        public double getHeight();

        public double getAngleStart();

        public double getAngleExtent();

        public boolean isEmpty();

        public void setArc(double x, double y, double w, double h, double angSt, double angExt, int closure);

        public void setAngleStart(double angSt);

        public void setAngleExtent(double angExt);

        protected Rectangle2D makeBounds(double x, double y, double w, double h);
    }

    protected Arc2D() {
    }

    protected Arc2D(int type) {
    }

    public abstract double getAngleStart();

    public abstract double getAngleExtent();

    public int getArcType();

    public Point2D getStartPoint();

    public Point2D getEndPoint();

    public abstract void setArc(double x, double y, double w, double h, double angSt, double angExt, int closure);

    public void setArc(Point2D loc, Dimension2D size, double angSt, double angExt, int closure);

    public void setArc(Rectangle2D rect, double angSt, double angExt, int closure);

    public void setArc(Arc2D a);

    public void setArcByCenter(double x, double y, double radius, double angSt, double angExt, int closure);

    public void setArcByTangent(Point2D p1, Point2D p2, Point2D p3, double radius);

    public abstract void setAngleStart(double angSt);

    public abstract void setAngleExtent(double angExt);

    public void setAngleStart(Point2D p);

    public void setAngles(double x1, double y1, double x2, double y2);

    public void setAngles(Point2D p1, Point2D p2);

    public void setArcType(int type);

    public void setFrame(double x, double y, double w, double h);

    public Rectangle2D getBounds2D();

    protected abstract Rectangle2D makeBounds(double x, double y, double w, double h);

    static double normalizeDegrees(double angle);

    public boolean containsAngle(double angle);

    public boolean contains(double x, double y);

    public boolean intersects(double x, double y, double w, double h);

    public boolean contains(double x, double y, double w, double h);

    public boolean contains(Rectangle2D r);

    public PathIterator getPathIterator(AffineTransform at);

    public int hashCode();

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object obj);
}
