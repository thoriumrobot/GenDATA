/*
 * Copyright (c) 1997, 2018, Oracle and/or its affiliates. All rights reserved.
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
import java.beans.ConstructorProperties;
import java.lang.annotation.Native;

public class BasicStroke implements Stroke {

    @Native
    public static final int JOIN_MITER;

    @Native
    public static final int JOIN_ROUND;

    @Native
    public static final int JOIN_BEVEL;

    @Native
    public static final int CAP_BUTT;

    @Native
    public static final int CAP_ROUND;

    @Native
    public static final int CAP_SQUARE;

    @ConstructorProperties({ "lineWidth", "endCap", "lineJoin", "miterLimit", "dashArray", "dashPhase" })
    public BasicStroke(float width, int cap, int join, float miterlimit, float[] dash, float dash_phase) {
    }

    public BasicStroke(float width, int cap, int join, float miterlimit) {
    }

    public BasicStroke(float width, int cap, int join) {
    }

    public BasicStroke(float width) {
    }

    public BasicStroke() {
    }

    public Shape createStrokedShape(Shape s);

    public float getLineWidth();

    public int getEndCap();

    public int getLineJoin();

    public float getMiterLimit();

    public float[] getDashArray();

    public float getDashPhase();

    public int hashCode();

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object obj);
}
