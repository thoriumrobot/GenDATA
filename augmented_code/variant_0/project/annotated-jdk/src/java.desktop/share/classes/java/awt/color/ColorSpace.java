/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 1997, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.awt.color;

    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.io.Serial;
    @Positive
import java.io.Serializable;
    @Positive
import java.lang.annotation.Native;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
public abstract class ColorSpace implements Serializable {

    @Positive
    private interface BuiltInSpace {

    @Positive
        ColorSpace SRGB;

    @Positive
        ColorSpace LRGB;

    @Positive
        ColorSpace XYZ;

    @Positive
        ColorSpace PYCC;

    @Positive
        ColorSpace GRAY;
    @Positive
    }

    @Positive
    @Native
    @Positive
    public static final int TYPE_XYZ;

    @Positive
    @Native
    @Positive
    public static final int TYPE_Lab;

    @Positive
    @Native
    @Positive
    public static final int TYPE_Luv;

    @Positive
    @Native
    @Positive
    public static final int TYPE_YCbCr;

    @Positive
    @Native
    @Positive
    public static final int TYPE_Yxy;

    @Positive
    @Native
    @Positive
    public static final int TYPE_RGB;

    @Positive
    @Native
    @Positive
    public static final int TYPE_GRAY;

    @Positive
    @Native
    @Positive
    public static final int TYPE_HSV;

    @Positive
    @Native
    @Positive
    public static final int TYPE_HLS;

    @Positive
    @Native
    @Positive
    public static final int TYPE_CMYK;

    @Positive
    @Native
    @Positive
    public static final int TYPE_CMY;

    @Positive
    @Native
    @Positive
    public static final int TYPE_2CLR;

    @Positive
    @Native
    @Positive
    public static final int TYPE_3CLR;

    @Positive
    @Native
    @Positive
    public static final int TYPE_4CLR;

    @Positive
    @Native
    @Positive
    public static final int TYPE_5CLR;

    @Positive
    @Native
    @Positive
    public static final int TYPE_6CLR;

    @Positive
    @Native
    @Positive
    public static final int TYPE_7CLR;

    @Positive
    @Native
    @Positive
    public static final int TYPE_8CLR;

    @Positive
    @Native
    @Positive
    public static final int TYPE_9CLR;

    @Positive
    @Native
    @Positive
    public static final int TYPE_ACLR;

    @Positive
    @Native
    @Positive
    public static final int TYPE_BCLR;

    @Positive
    @Native
    @Positive
    public static final int TYPE_CCLR;

    @Positive
    @Native
    @Positive
    public static final int TYPE_DCLR;

    @Positive
    @Native
    @Positive
    public static final int TYPE_ECLR;

    @Positive
    @Native
    @Positive
    public static final int TYPE_FCLR;

    @Positive
    @Native
    @Positive
    public static final int CS_sRGB;

    @Positive
    @Native
    @Positive
    public static final int CS_LINEAR_RGB;

    @Positive
    @Native
    @Positive
    public static final int CS_CIEXYZ;

    @Positive
    @Native
    @Positive
    public static final int CS_PYCC;

    @Positive
    @Native
    @Positive
    public static final int CS_GRAY;

    @Positive
    protected ColorSpace(int type, int numComponents) {
    @Positive
    }

    @Positive
    public static ColorSpace getInstance(int cspace);

    @Positive
    public boolean isCS_sRGB();

    @Positive
    public abstract float[] toRGB(float[] colorvalue);

    @Positive
    public abstract float[] fromRGB(float[] rgbvalue);

    @Positive
    public abstract float[] toCIEXYZ(float[] colorvalue);

    @Positive
    public abstract float[] fromCIEXYZ(float[] colorvalue);

    @Positive
    public int getType();

    @Positive
    public int getNumComponents();

    @Positive
    public String getName(int component);

    @Positive
    public float getMinValue(int component);

    @Positive
    public float getMaxValue(int component);

    @Positive
    final void rangeCheck(int component);
    @Positive
}
