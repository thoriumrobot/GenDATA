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
import java.awt.color.ColorSpace;
    @Positive
import java.awt.geom.AffineTransform;
    @Positive
import java.awt.geom.Rectangle2D;
    @Positive
import java.awt.image.ColorModel;
    @Positive
import java.beans.ConstructorProperties;
    @Positive
import java.io.Serial;

    @Positive
public class Color implements Paint, java.io.Serializable {

    @Positive
    public static final Color white;

    @Positive
    public static final Color WHITE;

    @Positive
    public static final Color lightGray;

    @Positive
    public static final Color LIGHT_GRAY;

    @Positive
    public static final Color gray;

    @Positive
    public static final Color GRAY;

    @Positive
    public static final Color darkGray;

    @Positive
    public static final Color DARK_GRAY;

    @Positive
    public static final Color black;

    @Positive
    public static final Color BLACK;

    @Positive
    public static final Color red;

    @Positive
    public static final Color RED;

    @Positive
    public static final Color pink;

    @Positive
    public static final Color PINK;

    @Positive
    public static final Color orange;

    @Positive
    public static final Color ORANGE;

    @Positive
    public static final Color yellow;

    @Positive
    public static final Color YELLOW;

    @Positive
    public static final Color green;

    @Positive
    public static final Color GREEN;

    @Positive
    public static final Color magenta;

    @Positive
    public static final Color MAGENTA;

    @Positive
    public static final Color cyan;

    @Positive
    public static final Color CYAN;

    @Positive
    public static final Color blue;

    @Positive
    public static final Color BLUE;

    @Positive
    public Color(int r, int g, int b) {
    @Positive
    }

    @Positive
    @ConstructorProperties({ "red", "green", "blue", "alpha" })
    @Positive
    public Color(int r, int g, int b, int a) {
    @Positive
    }

    @Positive
    public Color(int rgb) {
    @Positive
    }

    @Positive
    public Color(int rgba, boolean hasalpha) {
    @Positive
    }

    @Positive
    public Color(float r, float g, float b) {
    @Positive
    }

    @Positive
    public Color(float r, float g, float b, float a) {
    @Positive
    }

    @Positive
    public Color(ColorSpace cspace, float[] components, float alpha) {
    @Positive
    }

    @Positive
    public int getRed();

    @Positive
    public int getGreen();

    @Positive
    public int getBlue();

    @Positive
    public int getAlpha();

    @Positive
    public int getRGB();

    @Positive
    public Color brighter();

    @Positive
    public Color darker();

    @Positive
    public int hashCode();

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object obj);

    @Positive
    public String toString();

    @Positive
    public static Color decode(String nm) throws NumberFormatException;

    @Positive
    public static Color getColor(String nm);

    @Positive
    public static Color getColor(String nm, Color v);

    @Positive
    public static Color getColor(String nm, int v);

    @Positive
    public static int HSBtoRGB(float hue, float saturation, float brightness);

    @Positive
    public static float[] RGBtoHSB(int r, int g, int b, float[] hsbvals);

    @Positive
    public static Color getHSBColor(float h, float s, float b);

    @Positive
    public float[] getRGBComponents(float[] compArray);

    @Positive
    public float[] getRGBColorComponents(float[] compArray);

    @Positive
    public float[] getComponents(float[] compArray);

    @Positive
    public float[] getColorComponents(float[] compArray);

    @Positive
    public float[] getComponents(ColorSpace cspace, float[] compArray);

    @Positive
    public float[] getColorComponents(ColorSpace cspace, float[] compArray);

    @Positive
    public ColorSpace getColorSpace();

    @Positive
    public synchronized PaintContext createContext(ColorModel cm, Rectangle r, Rectangle2D r2d, AffineTransform xform, RenderingHints hints);

    @Positive
    public int getTransparency();
    @Positive
}

// CFWR semantic augmentation - variant 0
