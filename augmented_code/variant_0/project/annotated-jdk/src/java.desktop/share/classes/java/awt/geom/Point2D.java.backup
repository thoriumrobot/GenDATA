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

public abstract class Point2D implements Cloneable {

    public static class Float extends Point2D implements Serializable {

        public float x;

        public float y;

        public Float() {
        }

        public Float(float x, float y) {
        }

        public double getX();

        public double getY();

        public void setLocation(double x, double y);

        public void setLocation(float x, float y);

        public String toString();
    }

    public static class Double extends Point2D implements Serializable {

        public double x;

        public double y;

        public Double() {
        }

        public Double(double x, double y) {
        }

        public double getX();

        public double getY();

        public void setLocation(double x, double y);

        public String toString();
    }

    protected Point2D() {
    }

    public abstract double getX();

    public abstract double getY();

    public abstract void setLocation(double x, double y);

    public void setLocation(Point2D p);

    public static double distanceSq(double x1, double y1, double x2, double y2);

    public static double distance(double x1, double y1, double x2, double y2);

    public double distanceSq(double px, double py);

    public double distanceSq(Point2D pt);

    public double distance(double px, double py);

    public double distance(Point2D pt);

    public Object clone();

    public int hashCode();

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object obj);
}
