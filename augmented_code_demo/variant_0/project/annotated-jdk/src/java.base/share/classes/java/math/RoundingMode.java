/*
    @Positive
 * Copyright (c) 2003, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.math;

    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;

    @Positive
@SuppressWarnings("deprecation")
    @Positive
@AnnotatedFor("nullness")
    @Positive
public enum RoundingMode {

    @Positive
    UP(BigDecimal.ROUND_UP),
    @Positive
    DOWN(BigDecimal.ROUND_DOWN),
    @Positive
    CEILING(BigDecimal.ROUND_CEILING),
    @Positive
    FLOOR(BigDecimal.ROUND_FLOOR),
    @Positive
    HALF_UP(BigDecimal.ROUND_HALF_UP),
    @Positive
    HALF_DOWN(BigDecimal.ROUND_HALF_DOWN),
    @Positive
    HALF_EVEN(BigDecimal.ROUND_HALF_EVEN),
    @Positive
    UNNECESSARY(BigDecimal.ROUND_UNNECESSARY);

    @Positive
    public static RoundingMode valueOf(int rm);
    @Positive
}

// CFWR semantic augmentation - variant 0
