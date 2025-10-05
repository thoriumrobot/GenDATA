/*
    @Positive
 * Copyright (c) 1997, 2017, Oracle and/or its affiliates. All rights reserved.
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
import java.awt.image.ColorModel;
    @Positive
import java.lang.annotation.Native;
    @Positive
import sun.java2d.SunCompositeContext;

    @Positive
public final class AlphaComposite implements Composite {

    @Positive
    @Native
    @Positive
    public static final int CLEAR;

    @Positive
    @Native
    @Positive
    public static final int SRC;

    @Positive
    @Native
    @Positive
    public static final int DST;

    @Positive
    @Native
    @Positive
    public static final int SRC_OVER;

    @Positive
    @Native
    @Positive
    public static final int DST_OVER;

    @Positive
    @Native
    @Positive
    public static final int SRC_IN;

    @Positive
    @Native
    @Positive
    public static final int DST_IN;

    @Positive
    @Native
    @Positive
    public static final int SRC_OUT;

    @Positive
    @Native
    @Positive
    public static final int DST_OUT;

    @Positive
    @Native
    @Positive
    public static final int SRC_ATOP;

    @Positive
    @Native
    @Positive
    public static final int DST_ATOP;

    @Positive
    @Native
    @Positive
    public static final int XOR;

    @Positive
    public static final AlphaComposite Clear;

    @Positive
    public static final AlphaComposite Src;

    @Positive
    public static final AlphaComposite Dst;

    @Positive
    public static final AlphaComposite SrcOver;

    @Positive
    public static final AlphaComposite DstOver;

    @Positive
    public static final AlphaComposite SrcIn;

    @Positive
    public static final AlphaComposite DstIn;

    @Positive
    public static final AlphaComposite SrcOut;

    @Positive
    public static final AlphaComposite DstOut;

    @Positive
    public static final AlphaComposite SrcAtop;

    @Positive
    public static final AlphaComposite DstAtop;

    @Positive
    public static final AlphaComposite Xor;

    @Positive
    public static AlphaComposite getInstance(int rule);

    @Positive
    public static AlphaComposite getInstance(int rule, float alpha);

    @Positive
    public CompositeContext createContext(ColorModel srcColorModel, ColorModel dstColorModel, RenderingHints hints);

    @Positive
    public float getAlpha();

    @Positive
    public int getRule();

    @Positive
    public AlphaComposite derive(int rule);

    @Positive
    public AlphaComposite derive(float alpha);

    @Positive
    public int hashCode();

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object obj);
    @Positive
}

// CFWR semantic augmentation - variant 0
