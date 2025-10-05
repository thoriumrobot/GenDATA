/*
 * Copyright (c) 1999, 2016, Oracle and/or its affiliates. All rights reserved.
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
package sun.java2d.loops;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import java.awt.AlphaComposite;
import java.util.HashMap;

public final class CompositeType {

    public static final String DESC_ANY;

    public static final String DESC_XOR;

    public static final String DESC_CLEAR;

    public static final String DESC_SRC;

    public static final String DESC_DST;

    public static final String DESC_SRC_OVER;

    public static final String DESC_DST_OVER;

    public static final String DESC_SRC_IN;

    public static final String DESC_DST_IN;

    public static final String DESC_SRC_OUT;

    public static final String DESC_DST_OUT;

    public static final String DESC_SRC_ATOP;

    public static final String DESC_DST_ATOP;

    public static final String DESC_ALPHA_XOR;

    public static final String DESC_SRC_NO_EA;

    public static final String DESC_SRC_OVER_NO_EA;

    public static final String DESC_ANY_ALPHA;

    public static final CompositeType Any;

    public static final CompositeType General;

    public static final CompositeType AnyAlpha;

    public static final CompositeType Xor;

    public static final CompositeType Clear;

    public static final CompositeType Src;

    public static final CompositeType Dst;

    public static final CompositeType SrcOver;

    public static final CompositeType DstOver;

    public static final CompositeType SrcIn;

    public static final CompositeType DstIn;

    public static final CompositeType SrcOut;

    public static final CompositeType DstOut;

    public static final CompositeType SrcAtop;

    public static final CompositeType DstAtop;

    public static final CompositeType AlphaXor;

    public static final CompositeType SrcNoEa;

    public static final CompositeType SrcOverNoEa;

    public static final CompositeType OpaqueSrcOverNoEa;

    public CompositeType deriveSubType(String desc);

    public static CompositeType forAlphaComposite(AlphaComposite ac);

    public static synchronized int makeUniqueID(String desc);

    public int getUniqueID();

    public String getDescriptor();

    public CompositeType getSuperType();

    public int hashCode();

    public boolean isDerivedFrom(CompositeType other);

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object o);

    public String toString();
}
