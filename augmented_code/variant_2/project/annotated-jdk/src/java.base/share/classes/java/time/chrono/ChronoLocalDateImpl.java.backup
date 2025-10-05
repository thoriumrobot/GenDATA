/*
 * Copyright (c) 2012, 2019, Oracle and/or its affiliates. All rights reserved.
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
package java.time.chrono;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import static java.time.temporal.ChronoField.DAY_OF_MONTH;
import static java.time.temporal.ChronoField.ERA;
import static java.time.temporal.ChronoField.MONTH_OF_YEAR;
import static java.time.temporal.ChronoField.PROLEPTIC_MONTH;
import static java.time.temporal.ChronoField.YEAR_OF_ERA;
import java.io.Serializable;
import java.time.DateTimeException;
import java.time.temporal.ChronoUnit;
import java.time.temporal.Temporal;
import java.time.temporal.TemporalAdjuster;
import java.time.temporal.TemporalAmount;
import java.time.temporal.TemporalField;
import java.time.temporal.TemporalUnit;
import java.time.temporal.UnsupportedTemporalTypeException;
import java.time.temporal.ValueRange;
import java.util.Objects;

abstract class ChronoLocalDateImpl<D extends ChronoLocalDate> implements ChronoLocalDate, Temporal, TemporalAdjuster, Serializable {

    static <D extends ChronoLocalDate> D ensureValid(Chronology chrono, Temporal temporal);

    @Override
    @SuppressWarnings("unchecked")
    public D with(TemporalAdjuster adjuster);

    @Override
    @SuppressWarnings("unchecked")
    public D with(TemporalField field, long value);

    @Override
    @SuppressWarnings("unchecked")
    public D plus(TemporalAmount amount);

    @Override
    @SuppressWarnings("unchecked")
    public D plus(long amountToAdd, TemporalUnit unit);

    @Override
    @SuppressWarnings("unchecked")
    public D minus(TemporalAmount amount);

    @Override
    @SuppressWarnings("unchecked")
    public D minus(long amountToSubtract, TemporalUnit unit);

    abstract D plusYears(long yearsToAdd);

    abstract D plusMonths(long monthsToAdd);

    D plusWeeks(long weeksToAdd);

    abstract D plusDays(long daysToAdd);

    @SuppressWarnings("unchecked")
    D minusYears(long yearsToSubtract);

    @SuppressWarnings("unchecked")
    D minusMonths(long monthsToSubtract);

    @SuppressWarnings("unchecked")
    D minusWeeks(long weeksToSubtract);

    @SuppressWarnings("unchecked")
    D minusDays(long daysToSubtract);

    @Override
    public long until(Temporal endExclusive, TemporalUnit unit);

    @Override
    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object obj);

    @Override
    public int hashCode();

    @Override
    public String toString();
}
