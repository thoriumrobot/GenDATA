/*
 * Copyright (c) 1995, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.awt;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import java.awt.geom.Rectangle2D;
import java.beans.Transient;
import java.io.Serial;

public class Rectangle extends Rectangle2D implements Shape, java.io.Serializable {

    public int x;

    public int y;

    public int width;

    public int height;

    public Rectangle() {
    }

    public Rectangle(Rectangle r) {
    }

    public Rectangle(int x, int y, int width, int height) {
    }

    public Rectangle(int width, int height) {
    }

    public Rectangle(Point p, Dimension d) {
    }

    public Rectangle(Point p) {
    }

    public Rectangle(Dimension d) {
    }

    public double getX();

    public double getY();

    public double getWidth();

    public double getHeight();

    @Transient
    public Rectangle getBounds();

    public Rectangle2D getBounds2D();

    public void setBounds(Rectangle r);

    public void setBounds(int x, int y, int width, int height);

    public void setRect(double x, double y, double width, double height);

    @Deprecated
    public void reshape(int x, int y, int width, int height);

    public Point getLocation();

    public void setLocation(Point p);

    public void setLocation(int x, int y);

    @Deprecated
    public void move(int x, int y);

    public void translate(int dx, int dy);

    public Dimension getSize();

    public void setSize(Dimension d);

    public void setSize(int width, int height);

    @Deprecated
    public void resize(int width, int height);

    public boolean contains(Point p);

    public boolean contains(int x, int y);

    public boolean contains(Rectangle r);

    public boolean contains(int X, int Y, int W, int H);

    @Deprecated
    public boolean inside(int X, int Y);

    public boolean intersects(Rectangle r);

    public Rectangle intersection(Rectangle r);

    public Rectangle union(Rectangle r);

    public void add(int newx, int newy);

    public void add(Point pt);

    public void add(Rectangle r);

    public void grow(int h, int v);

    public boolean isEmpty();

    public int outcode(double x, double y);

    public Rectangle2D createIntersection(Rectangle2D r);

    public Rectangle2D createUnion(Rectangle2D r);

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object obj);

    public String toString();
}
