/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 1997, 2013, Oracle and/or its affiliates. All rights reserved.
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
import java.awt.geom.Rectangle2D;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
public final class GlyphMetrics {

    @Positive
    public static final byte STANDARD;

    @Positive
    public static final byte LIGATURE;

    @Positive
    public static final byte COMBINING;

    @Positive
    public static final byte COMPONENT;

    @Positive
    public static final byte WHITESPACE;

    @Positive
    public GlyphMetrics(float advance, Rectangle2D bounds, byte glyphType) {
    @Positive
    }

    @Positive
    public GlyphMetrics(boolean horizontal, float advanceX, float advanceY, Rectangle2D bounds, byte glyphType) {
    @Positive
    }

    @Positive
    public float getAdvance();

    @Positive
    public float getAdvanceX();

    @Positive
    public float getAdvanceY();

    @Positive
    public Rectangle2D getBounds2D();

    @Positive
    public float getLSB();

    @Positive
    public float getRSB();

    @Positive
    public int getType();

    @Positive
    public boolean isStandard();

    @Positive
    public boolean isLigature();

    @Positive
    public boolean isCombining();

    @Positive
    public boolean isComponent();

    @Positive
    public boolean isWhitespace();
    @Positive
}
