/*
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
 *
 */
package sun.font;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import java.awt.font.LineMetrics;
import java.awt.font.GraphicAttribute;

public final class CoreMetrics {

    public CoreMetrics(float ascent, float descent, float leading, float height, int baselineIndex, float[] baselineOffsets, float strikethroughOffset, float strikethroughThickness, float underlineOffset, float underlineThickness, float ssOffset, float italicAngle) {
    }

    public static CoreMetrics get(LineMetrics lm);

    public int hashCode();

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object rhs);

    public boolean equals(CoreMetrics rhs);

    public float effectiveBaselineOffset(float[] fullOffsets);

    public final float ascent;

    public final float descent;

    public final float leading;

    public final float height;

    public final int baselineIndex;

    public final float[] baselineOffsets;

    public final float strikethroughOffset;

    public final float strikethroughThickness;

    public final float underlineOffset;

    public final float underlineThickness;

    public final float ssOffset;

    public final float italicAngle;
}
