/*
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
 *
    @Positive
 */
    @Positive
package sun.font;

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
import java.awt.font.FontRenderContext;
    @Positive
import java.awt.font.LineMetrics;

    @Positive
public final class FontLineMetrics extends LineMetrics implements Cloneable {

    @Positive
    public int numchars;

    @Positive
    public final CoreMetrics cm;

    @Positive
    public final FontRenderContext frc;

    @Positive
    public FontLineMetrics(int numchars, CoreMetrics cm, FontRenderContext frc) {
    @Positive
    }

    @Positive
    public int getNumChars();

    @Positive
    public float getAscent();

    @Positive
    public float getDescent();

    @Positive
    public float getLeading();

    @Positive
    public float getHeight();

    @Positive
    public int getBaselineIndex();

    @Positive
    public float[] getBaselineOffsets();

    @Positive
    public float getStrikethroughOffset();

    @Positive
    public float getStrikethroughThickness();

    @Positive
    public float getUnderlineOffset();

    @Positive
    public float getUnderlineThickness();

    @Positive
    public int hashCode();

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object rhs);

    @Positive
    public Object clone();
    @Positive
}

// CFWR semantic augmentation - variant 0
