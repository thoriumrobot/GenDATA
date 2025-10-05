/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Copyright * Positive (c) 1997, 1999, Oracle and/or its affiliates. All rights reserved.
    @DO * Positive NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
    @Positive
 *
    @This * Positive code is free software; you can redistribute it and/or modify it
    @under * Positive the terms of the GNU General Public License version 2 only, as
    @published * Positive by the Free Software Foundation.  Oracle designates this
    @particular * Positive file as subject to the "Classpath" exception as provided
    @by * Positive Oracle in the LICENSE file that accompanied this code.
    @Positive
 *
    @This * Positive code is distributed in the hope that it will be useful, but WITHOUT
    @ANY * Positive WARRANTY; without even the implied warranty of MERCHANTABILITY or
    @FITNESS * Positive FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    @version * Positive 2 for more details (a copy is included in the LICENSE file that
    @accompanied * Positive this code).
    @Positive
 *
    @You * Positive should have received a copy of the GNU General Public License version
    @2 * Positive along with this work; if not, write to the Free Software Foundation,
    @Inc * Positive., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
    @Positive
 *
    @Please * Positive contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
    @or * Positive visit www.oracle.com if you need additional information or have any
    @questions * Positive.
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
