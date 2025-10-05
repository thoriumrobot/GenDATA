/*
 * Copyright (c) 2006, 2021, Oracle and/or its affiliates. All rights reserved.
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
import java.io.IOException;
import java.io.Serial;
import java.io.Serializable;
import java.io.StreamCorruptedException;
import java.util.Arrays;
import sun.awt.geom.Curve;

@AnnotatedFor({ "interning" })
@UsesObjectEquals
public abstract class Path2D implements Shape, Cloneable {

    public static final int WIND_EVEN_ODD;

    public static final int WIND_NON_ZERO;

    abstract float[] cloneCoordsFloat(AffineTransform at);

    abstract double[] cloneCoordsDouble(AffineTransform at);

    abstract void append(float x, float y);

    abstract void append(double x, double y);

    abstract Point2D getPoint(int coordindex);

    abstract void needRoom(boolean needMove, int newCoords);

    abstract int pointCrossings(double px, double py);

    abstract int rectCrossings(double rxmin, double rymin, double rxmax, double rymax);

    static byte[] expandPointTypes(byte[] oldPointTypes, int needed);

    public static class Float extends Path2D implements Serializable {

        public Float() {
        }

        public Float(int rule) {
        }

        public Float(int rule, int initialCapacity) {
        }

        public Float(Shape s) {
        }

        public Float(Shape s, AffineTransform at) {
        }

        @Override
        public final void trimToSize();

        @Override
        float[] cloneCoordsFloat(AffineTransform at);

        @Override
        double[] cloneCoordsDouble(AffineTransform at);

        void append(float x, float y);

        void append(double x, double y);

        Point2D getPoint(int coordindex);

        @Override
        void needRoom(boolean needMove, int newCoords);

        static float[] expandCoords(float[] oldCoords, int needed);

        public final synchronized void moveTo(double x, double y);

        public final synchronized void moveTo(float x, float y);

        public final synchronized void lineTo(double x, double y);

        public final synchronized void lineTo(float x, float y);

        public final synchronized void quadTo(double x1, double y1, double x2, double y2);

        public final synchronized void quadTo(float x1, float y1, float x2, float y2);

        public final synchronized void curveTo(double x1, double y1, double x2, double y2, double x3, double y3);

        public final synchronized void curveTo(float x1, float y1, float x2, float y2, float x3, float y3);

        int pointCrossings(double px, double py);

        int rectCrossings(double rxmin, double rymin, double rxmax, double rymax);

        public final void append(PathIterator pi, boolean connect);

        public final void transform(AffineTransform at);

        public final synchronized Rectangle2D getBounds2D();

        public final PathIterator getPathIterator(AffineTransform at);

        public final Object clone();

        static class CopyIterator extends Path2D.Iterator {

            public int currentSegment(float[] coords);

            public int currentSegment(double[] coords);
        }

        static class TxIterator extends Path2D.Iterator {

            public int currentSegment(float[] coords);

            public int currentSegment(double[] coords);
        }
    }

    public static class Double extends Path2D implements Serializable {

        public Double() {
        }

        public Double(int rule) {
        }

        public Double(int rule, int initialCapacity) {
        }

        public Double(Shape s) {
        }

        public Double(Shape s, AffineTransform at) {
        }

        @Override
        public final void trimToSize();

        @Override
        float[] cloneCoordsFloat(AffineTransform at);

        @Override
        double[] cloneCoordsDouble(AffineTransform at);

        void append(float x, float y);

        void append(double x, double y);

        Point2D getPoint(int coordindex);

        @Override
        void needRoom(boolean needMove, int newCoords);

        static double[] expandCoords(double[] oldCoords, int needed);

        public final synchronized void moveTo(double x, double y);

        public final synchronized void lineTo(double x, double y);

        public final synchronized void quadTo(double x1, double y1, double x2, double y2);

        public final synchronized void curveTo(double x1, double y1, double x2, double y2, double x3, double y3);

        int pointCrossings(double px, double py);

        int rectCrossings(double rxmin, double rymin, double rxmax, double rymax);

        public final void append(PathIterator pi, boolean connect);

        public final void transform(AffineTransform at);

        public final synchronized Rectangle2D getBounds2D();

        public final PathIterator getPathIterator(AffineTransform at);

        public final Object clone();

        static class CopyIterator extends Path2D.Iterator {

            public int currentSegment(float[] coords);

            public int currentSegment(double[] coords);
        }

        static class TxIterator extends Path2D.Iterator {

            public int currentSegment(float[] coords);

            public int currentSegment(double[] coords);
        }
    }

    public abstract void moveTo(double x, double y);

    public abstract void lineTo(double x, double y);

    public abstract void quadTo(double x1, double y1, double x2, double y2);

    public abstract void curveTo(double x1, double y1, double x2, double y2, double x3, double y3);

    public final synchronized void closePath();

    public final void append(Shape s, boolean connect);

    public abstract void append(PathIterator pi, boolean connect);

    public final synchronized int getWindingRule();

    public final void setWindingRule(int rule);

    public final synchronized Point2D getCurrentPoint();

    public final synchronized void reset();

    public abstract void transform(AffineTransform at);

    public final synchronized Shape createTransformedShape(AffineTransform at);

    public final Rectangle getBounds();

    public static boolean contains(PathIterator pi, double x, double y);

    public static boolean contains(PathIterator pi, Point2D p);

    public final boolean contains(double x, double y);

    public final boolean contains(Point2D p);

    public static boolean contains(PathIterator pi, double x, double y, double w, double h);

    public static boolean contains(PathIterator pi, Rectangle2D r);

    public final boolean contains(double x, double y, double w, double h);

    public final boolean contains(Rectangle2D r);

    public static boolean intersects(PathIterator pi, double x, double y, double w, double h);

    public static boolean intersects(PathIterator pi, Rectangle2D r);

    public final boolean intersects(double x, double y, double w, double h);

    public final boolean intersects(Rectangle2D r);

    public final PathIterator getPathIterator(AffineTransform at, double flatness);

    public abstract Object clone();

    public abstract void trimToSize();

    final void writeObject(java.io.ObjectOutputStream s, boolean isdbl) throws java.io.IOException;

    final void readObject(java.io.ObjectInputStream s, boolean storedbl) throws java.lang.ClassNotFoundException, java.io.IOException;

    abstract static class Iterator implements PathIterator {

        public int getWindingRule();

        public boolean isDone();

        public void next();
    }
}
