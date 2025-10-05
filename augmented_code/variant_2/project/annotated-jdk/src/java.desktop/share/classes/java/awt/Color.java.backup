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
import java.awt.color.ColorSpace;
import java.awt.geom.AffineTransform;
import java.awt.geom.Rectangle2D;
import java.awt.image.ColorModel;
import java.beans.ConstructorProperties;
import java.io.Serial;

public class Color implements Paint, java.io.Serializable {

    public static final Color white;

    public static final Color WHITE;

    public static final Color lightGray;

    public static final Color LIGHT_GRAY;

    public static final Color gray;

    public static final Color GRAY;

    public static final Color darkGray;

    public static final Color DARK_GRAY;

    public static final Color black;

    public static final Color BLACK;

    public static final Color red;

    public static final Color RED;

    public static final Color pink;

    public static final Color PINK;

    public static final Color orange;

    public static final Color ORANGE;

    public static final Color yellow;

    public static final Color YELLOW;

    public static final Color green;

    public static final Color GREEN;

    public static final Color magenta;

    public static final Color MAGENTA;

    public static final Color cyan;

    public static final Color CYAN;

    public static final Color blue;

    public static final Color BLUE;

    public Color(int r, int g, int b) {
    }

    @ConstructorProperties({ "red", "green", "blue", "alpha" })
    public Color(int r, int g, int b, int a) {
    }

    public Color(int rgb) {
    }

    public Color(int rgba, boolean hasalpha) {
    }

    public Color(float r, float g, float b) {
    }

    public Color(float r, float g, float b, float a) {
    }

    public Color(ColorSpace cspace, float[] components, float alpha) {
    }

    public int getRed();

    public int getGreen();

    public int getBlue();

    public int getAlpha();

    public int getRGB();

    public Color brighter();

    public Color darker();

    public int hashCode();

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object obj);

    public String toString();

    public static Color decode(String nm) throws NumberFormatException;

    public static Color getColor(String nm);

    public static Color getColor(String nm, Color v);

    public static Color getColor(String nm, int v);

    public static int HSBtoRGB(float hue, float saturation, float brightness);

    public static float[] RGBtoHSB(int r, int g, int b, float[] hsbvals);

    public static Color getHSBColor(float h, float s, float b);

    public float[] getRGBComponents(float[] compArray);

    public float[] getRGBColorComponents(float[] compArray);

    public float[] getComponents(float[] compArray);

    public float[] getColorComponents(float[] compArray);

    public float[] getComponents(ColorSpace cspace, float[] compArray);

    public float[] getColorComponents(ColorSpace cspace, float[] compArray);

    public ColorSpace getColorSpace();

    public synchronized PaintContext createContext(ColorModel cm, Rectangle r, Rectangle2D r2d, AffineTransform xform, RenderingHints hints);

    public int getTransparency();
}
