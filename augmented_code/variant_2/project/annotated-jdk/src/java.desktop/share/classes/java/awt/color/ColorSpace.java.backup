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
package java.awt.color;

import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.io.Serial;
import java.io.Serializable;
import java.lang.annotation.Native;

@AnnotatedFor({ "interning" })
@UsesObjectEquals
public abstract class ColorSpace implements Serializable {

    private interface BuiltInSpace {

        ColorSpace SRGB;

        ColorSpace LRGB;

        ColorSpace XYZ;

        ColorSpace PYCC;

        ColorSpace GRAY;
    }

    @Native
    public static final int TYPE_XYZ;

    @Native
    public static final int TYPE_Lab;

    @Native
    public static final int TYPE_Luv;

    @Native
    public static final int TYPE_YCbCr;

    @Native
    public static final int TYPE_Yxy;

    @Native
    public static final int TYPE_RGB;

    @Native
    public static final int TYPE_GRAY;

    @Native
    public static final int TYPE_HSV;

    @Native
    public static final int TYPE_HLS;

    @Native
    public static final int TYPE_CMYK;

    @Native
    public static final int TYPE_CMY;

    @Native
    public static final int TYPE_2CLR;

    @Native
    public static final int TYPE_3CLR;

    @Native
    public static final int TYPE_4CLR;

    @Native
    public static final int TYPE_5CLR;

    @Native
    public static final int TYPE_6CLR;

    @Native
    public static final int TYPE_7CLR;

    @Native
    public static final int TYPE_8CLR;

    @Native
    public static final int TYPE_9CLR;

    @Native
    public static final int TYPE_ACLR;

    @Native
    public static final int TYPE_BCLR;

    @Native
    public static final int TYPE_CCLR;

    @Native
    public static final int TYPE_DCLR;

    @Native
    public static final int TYPE_ECLR;

    @Native
    public static final int TYPE_FCLR;

    @Native
    public static final int CS_sRGB;

    @Native
    public static final int CS_LINEAR_RGB;

    @Native
    public static final int CS_CIEXYZ;

    @Native
    public static final int CS_PYCC;

    @Native
    public static final int CS_GRAY;

    protected ColorSpace(int type, int numComponents) {
    }

    public static ColorSpace getInstance(int cspace);

    public boolean isCS_sRGB();

    public abstract float[] toRGB(float[] colorvalue);

    public abstract float[] fromRGB(float[] rgbvalue);

    public abstract float[] toCIEXYZ(float[] colorvalue);

    public abstract float[] fromCIEXYZ(float[] colorvalue);

    public int getType();

    public int getNumComponents();

    public String getName(int component);

    public float getMinValue(int component);

    public float getMaxValue(int component);

    final void rangeCheck(int component);
}
