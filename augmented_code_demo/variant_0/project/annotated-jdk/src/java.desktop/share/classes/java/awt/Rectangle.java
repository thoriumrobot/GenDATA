/*
    @Positive
 * Copyright (c) 1995, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.awt;

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
import java.awt.geom.Rectangle2D;
    @Positive
import java.beans.Transient;
    @Positive
import java.io.Serial;

    @Positive
public class Rectangle extends Rectangle2D implements Shape, java.io.Serializable {

    @Positive
    public int x;

    @Positive
    public int y;

    @Positive
    public int width;

    @Positive
    public int height;

    @Positive
    public Rectangle() {
    @Positive
    }

    @Positive
    public Rectangle(Rectangle r) {
    @Positive
    }

    @Positive
    public Rectangle(int x, int y, int width, int height) {
    @Positive
    }

    @Positive
    public Rectangle(int width, int height) {
    @Positive
    }

    @Positive
    public Rectangle(Point p, Dimension d) {
    @Positive
    }

    @Positive
    public Rectangle(Point p) {
    @Positive
    }

    @Positive
    public Rectangle(Dimension d) {
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
    @Transient
    @Positive
    public Rectangle getBounds();

    @Positive
    public Rectangle2D getBounds2D();

    @Positive
    public void setBounds(Rectangle r);

    @Positive
    public void setBounds(int x, int y, int width, int height);

    @Positive
    public void setRect(double x, double y, double width, double height);

    @Positive
    @Deprecated
    @Positive
    public void reshape(int x, int y, int width, int height);

    @Positive
    public Point getLocation();

    @Positive
    public void setLocation(Point p);

    @Positive
    public void setLocation(int x, int y);

    @Positive
    @Deprecated
    @Positive
    public void move(int x, int y);

    @Positive
    public void translate(int dx, int dy);

    @Positive
    public Dimension getSize();

    @Positive
    public void setSize(Dimension d);

    @Positive
    public void setSize(int width, int height);

    @Positive
    @Deprecated
    @Positive
    public void resize(int width, int height);

    @Positive
    public boolean contains(Point p);

    @Positive
    public boolean contains(int x, int y);

    @Positive
    public boolean contains(Rectangle r);

    @Positive
    public boolean contains(int X, int Y, int W, int H);

    @Positive
    @Deprecated
    @Positive
    public boolean inside(int X, int Y);

    @Positive
    public boolean intersects(Rectangle r);

    @Positive
    public Rectangle intersection(Rectangle r);

    @Positive
    public Rectangle union(Rectangle r);

    @Positive
    public void add(int newx, int newy);

    @Positive
    public void add(Point pt);

    @Positive
    public void add(Rectangle r);

    @Positive
    public void grow(int h, int v);

    @Positive
    public boolean isEmpty();

    @Positive
    public int outcode(double x, double y);

    @Positive
    public Rectangle2D createIntersection(Rectangle2D r);

    @Positive
    public Rectangle2D createUnion(Rectangle2D r);

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object obj);

    @Positive
    public String toString();
    @Positive
}

// CFWR semantic augmentation - variant 0
