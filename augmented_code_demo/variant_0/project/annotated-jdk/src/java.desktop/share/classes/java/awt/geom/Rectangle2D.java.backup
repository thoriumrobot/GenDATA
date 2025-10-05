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
import java.io.Serial;
import java.io.Serializable;

public abstract class Rectangle2D extends RectangularShape {

    public static final int OUT_LEFT;

    public static final int OUT_TOP;

    public static final int OUT_RIGHT;

    public static final int OUT_BOTTOM;

    public static class Float extends Rectangle2D implements Serializable {

        public float x;

        public float y;

        public float width;

        public float height;

        public Float() {
        }

        public Float(float x, float y, float w, float h) {
        }

        public double getX();

        public double getY();

        public double getWidth();

        public double getHeight();

        public boolean isEmpty();

        public void setRect(float x, float y, float w, float h);

        public void setRect(double x, double y, double w, double h);

        public void setRect(Rectangle2D r);

        public int outcode(double x, double y);

        public Rectangle2D getBounds2D();

        public Rectangle2D createIntersection(Rectangle2D r);

        public Rectangle2D createUnion(Rectangle2D r);

        public String toString();
    }

    public static class Double extends Rectangle2D implements Serializable {

        public double x;

        public double y;

        public double width;

        public double height;

        public Double() {
        }

        public Double(double x, double y, double w, double h) {
        }

        public double getX();

        public double getY();

        public double getWidth();

        public double getHeight();

        public boolean isEmpty();

        public void setRect(double x, double y, double w, double h);

        public void setRect(Rectangle2D r);

        public int outcode(double x, double y);

        public Rectangle2D getBounds2D();

        public Rectangle2D createIntersection(Rectangle2D r);

        public Rectangle2D createUnion(Rectangle2D r);

        public String toString();
    }

    protected Rectangle2D() {
    }

    public abstract void setRect(double x, double y, double w, double h);

    public void setRect(Rectangle2D r);

    public boolean intersectsLine(double x1, double y1, double x2, double y2);

    public boolean intersectsLine(Line2D l);

    public abstract int outcode(double x, double y);

    public int outcode(Point2D p);

    public void setFrame(double x, double y, double w, double h);

    public Rectangle2D getBounds2D();

    public boolean contains(double x, double y);

    public boolean intersects(double x, double y, double w, double h);

    public boolean contains(double x, double y, double w, double h);

    public abstract Rectangle2D createIntersection(Rectangle2D r);

    public static void intersect(Rectangle2D src1, Rectangle2D src2, Rectangle2D dest);

    public abstract Rectangle2D createUnion(Rectangle2D r);

    public static void union(Rectangle2D src1, Rectangle2D src2, Rectangle2D dest);

    public void add(double newx, double newy);

    public void add(Point2D pt);

    public void add(Rectangle2D r);

    public PathIterator getPathIterator(AffineTransform at);

    public PathIterator getPathIterator(AffineTransform at, double flatness);

    public int hashCode();

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object obj);
}
