/*
 * Copyright (c) 1997, 2010, Oracle and/or its affiliates. All rights reserved.
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

import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.awt.geom.Point2D;
import java.awt.geom.Rectangle2D;
import java.awt.geom.AffineTransform;
import java.awt.image.ColorModel;
import java.beans.ConstructorProperties;

@AnnotatedFor({ "interning" })
@UsesObjectEquals
public class GradientPaint implements Paint {

    public GradientPaint(float x1, float y1, Color color1, float x2, float y2, Color color2) {
    }

    public GradientPaint(Point2D pt1, Color color1, Point2D pt2, Color color2) {
    }

    public GradientPaint(float x1, float y1, Color color1, float x2, float y2, Color color2, boolean cyclic) {
    }

    @ConstructorProperties({ "point1", "color1", "point2", "color2", "cyclic" })
    public GradientPaint(Point2D pt1, Color color1, Point2D pt2, Color color2, boolean cyclic) {
    }

    public Point2D getPoint1();

    public Color getColor1();

    public Point2D getPoint2();

    public Color getColor2();

    public boolean isCyclic();

    public PaintContext createContext(ColorModel cm, Rectangle deviceBounds, Rectangle2D userBounds, AffineTransform xform, RenderingHints hints);

    public int getTransparency();
}
