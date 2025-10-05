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
public abstract class Line2D implements Shape, Cloneable {

    public static class Float extends Line2D implements Serializable {

        public float x1;

        public float y1;

        public float x2;

        public float y2;

        public Float() {
        }

        public Float(float x1, float y1, float x2, float y2) {
        }

        public Float(Point2D p1, Point2D p2) {
        }

        public double getX1();

        public double getY1();

        public Point2D getP1();

        public double getX2();

        public double getY2();

        public Point2D getP2();

        public void setLine(double x1, double y1, double x2, double y2);

        public void setLine(float x1, float y1, float x2, float y2);

        public Rectangle2D getBounds2D();
    }

    public static class Double extends Line2D implements Serializable {

        public double x1;

        public double y1;

        public double x2;

        public double y2;

        public Double() {
        }

        public Double(double x1, double y1, double x2, double y2) {
        }

        public Double(Point2D p1, Point2D p2) {
        }

        public double getX1();

        public double getY1();

        public Point2D getP1();

        public double getX2();

        public double getY2();

        public Point2D getP2();

        public void setLine(double x1, double y1, double x2, double y2);

        public Rectangle2D getBounds2D();
    }

    protected Line2D() {
    }

    public abstract double getX1();

    public abstract double getY1();

    public abstract Point2D getP1();

    public abstract double getX2();

    public abstract double getY2();

    public abstract Point2D getP2();

    public abstract void setLine(double x1, double y1, double x2, double y2);

    public void setLine(Point2D p1, Point2D p2);

    public void setLine(Line2D l);

    public static int relativeCCW(double x1, double y1, double x2, double y2, double px, double py);

    public int relativeCCW(double px, double py);

    public int relativeCCW(Point2D p);

    public static boolean linesIntersect(double x1, double y1, double x2, double y2, double x3, double y3, double x4, double y4);

    public boolean intersectsLine(double x1, double y1, double x2, double y2);

    public boolean intersectsLine(Line2D l);

    public static double ptSegDistSq(double x1, double y1, double x2, double y2, double px, double py);

    public static double ptSegDist(double x1, double y1, double x2, double y2, double px, double py);

    public double ptSegDistSq(double px, double py);

    public double ptSegDistSq(Point2D pt);

    public double ptSegDist(double px, double py);

    public double ptSegDist(Point2D pt);

    public static double ptLineDistSq(double x1, double y1, double x2, double y2, double px, double py);

    public static double ptLineDist(double x1, double y1, double x2, double y2, double px, double py);

    public double ptLineDistSq(double px, double py);

    public double ptLineDistSq(Point2D pt);

    public double ptLineDist(double px, double py);

    public double ptLineDist(Point2D pt);

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
