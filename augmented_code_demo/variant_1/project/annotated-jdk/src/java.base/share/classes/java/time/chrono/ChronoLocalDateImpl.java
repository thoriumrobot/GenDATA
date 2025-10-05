/*
    @Positive
 * Copyright (c) 2012, 2019, Oracle and/or its affiliates. All rights reserved.
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
package java.time.chrono;

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
import static java.time.temporal.ChronoField.DAY_OF_MONTH;
    @Positive
import static java.time.temporal.ChronoField.ERA;
    @Positive
import static java.time.temporal.ChronoField.MONTH_OF_YEAR;
    @Positive
import static java.time.temporal.ChronoField.PROLEPTIC_MONTH;
    @Positive
import static java.time.temporal.ChronoField.YEAR_OF_ERA;
    @Positive
import java.io.Serializable;
    @Positive
import java.time.DateTimeException;
    @Positive
import java.time.temporal.ChronoUnit;
    @Positive
import java.time.temporal.Temporal;
    @Positive
import java.time.temporal.TemporalAdjuster;
    @Positive
import java.time.temporal.TemporalAmount;
    @Positive
import java.time.temporal.TemporalField;
    @Positive
import java.time.temporal.TemporalUnit;
    @Positive
import java.time.temporal.UnsupportedTemporalTypeException;
    @Positive
import java.time.temporal.ValueRange;
    @Positive
import java.util.Objects;

    @Positive
abstract class ChronoLocalDateImpl<D extends ChronoLocalDate> implements ChronoLocalDate, Temporal, TemporalAdjuster, Serializable {

    @Positive
    static <D extends ChronoLocalDate> D ensureValid(Chronology chrono, Temporal temporal);

    @Positive
    @Override
    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    public D with(TemporalAdjuster adjuster);

    @Positive
    @Override
    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    public D with(TemporalField field, long value);

    @Positive
    @Override
    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    public D plus(TemporalAmount amount);

    @Positive
    @Override
    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    public D plus(long amountToAdd, TemporalUnit unit);

    @Positive
    @Override
    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    public D minus(TemporalAmount amount);

    @Positive
    @Override
    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    public D minus(long amountToSubtract, TemporalUnit unit);

    @Positive
    abstract D plusYears(long yearsToAdd);

    @Positive
    abstract D plusMonths(long monthsToAdd);

    @Positive
    D plusWeeks(long weeksToAdd);

    @Positive
    abstract D plusDays(long daysToAdd);

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    D minusYears(long yearsToSubtract);

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    D minusMonths(long monthsToSubtract);

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    D minusWeeks(long weeksToSubtract);

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    D minusDays(long daysToSubtract);

    @Positive
    @Override
    @Positive
    public long until(Temporal endExclusive, TemporalUnit unit);

    @Positive
    @Override
    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object obj);

    @Positive
    @Override
    @Positive
    public int hashCode();

    @Positive
    @Override
    @Positive
    public String toString();
    @Positive
}

// CFWR semantic augmentation - variant 1
