/*
    @Positive
 * Copyright (c) 1997, 1999, Oracle and/or its affiliates. All rights reserved.
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
package java.awt.font;

    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
public final class GlyphJustificationInfo {

    @Positive
    public GlyphJustificationInfo(float weight, boolean growAbsorb, int growPriority, float growLeftLimit, float growRightLimit, boolean shrinkAbsorb, int shrinkPriority, float shrinkLeftLimit, float shrinkRightLimit) {
    @Positive
    }

    @Positive
    public static final int PRIORITY_KASHIDA;

    @Positive
    public static final int PRIORITY_WHITESPACE;

    @Positive
    public static final int PRIORITY_INTERCHAR;

    @Positive
    public static final int PRIORITY_NONE;

    @Positive
    public final float weight;

    @Positive
    public final int growPriority;

    @Positive
    public final boolean growAbsorb;

    @Positive
    public final float growLeftLimit;

    @Positive
    public final float growRightLimit;

    @Positive
    public final int shrinkPriority;

    @Positive
    public final boolean shrinkAbsorb;

    @Positive
    public final float shrinkLeftLimit;

    @Positive
    public final float shrinkRightLimit;
    @Positive
}

// CFWR semantic augmentation - variant 0
