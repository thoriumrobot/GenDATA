/*
 * Copyright (c) 1997, 2017, Oracle and/or its affiliates. All rights reserved.
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
import java.awt.image.ColorModel;
import java.lang.annotation.Native;
import sun.java2d.SunCompositeContext;

public final class AlphaComposite implements Composite {

    @Native
    public static final int CLEAR;

    @Native
    public static final int SRC;

    @Native
    public static final int DST;

    @Native
    public static final int SRC_OVER;

    @Native
    public static final int DST_OVER;

    @Native
    public static final int SRC_IN;

    @Native
    public static final int DST_IN;

    @Native
    public static final int SRC_OUT;

    @Native
    public static final int DST_OUT;

    @Native
    public static final int SRC_ATOP;

    @Native
    public static final int DST_ATOP;

    @Native
    public static final int XOR;

    public static final AlphaComposite Clear;

    public static final AlphaComposite Src;

    public static final AlphaComposite Dst;

    public static final AlphaComposite SrcOver;

    public static final AlphaComposite DstOver;

    public static final AlphaComposite SrcIn;

    public static final AlphaComposite DstIn;

    public static final AlphaComposite SrcOut;

    public static final AlphaComposite DstOut;

    public static final AlphaComposite SrcAtop;

    public static final AlphaComposite DstAtop;

    public static final AlphaComposite Xor;

    public static AlphaComposite getInstance(int rule);

    public static AlphaComposite getInstance(int rule, float alpha);

    public CompositeContext createContext(ColorModel srcColorModel, ColorModel dstColorModel, RenderingHints hints);

    public float getAlpha();

    public int getRule();

    public AlphaComposite derive(int rule);

    public AlphaComposite derive(float alpha);

    public int hashCode();

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object obj);
}
