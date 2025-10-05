/*
 * Copyright (c) 2012, 2020, Oracle and/or its affiliates. All rights reserved.
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
package java.time;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import static java.time.temporal.ChronoUnit.DAYS;
import static java.time.temporal.ChronoUnit.MONTHS;
import static java.time.temporal.ChronoUnit.YEARS;
import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.io.InvalidObjectException;
import java.io.ObjectInputStream;
import java.io.Serializable;
import java.time.chrono.ChronoLocalDate;
import java.time.chrono.ChronoPeriod;
import java.time.chrono.Chronology;
import java.time.chrono.IsoChronology;
import java.time.format.DateTimeParseException;
import java.time.temporal.ChronoUnit;
import java.time.temporal.Temporal;
import java.time.temporal.TemporalAccessor;
import java.time.temporal.TemporalAmount;
import java.time.temporal.TemporalQueries;
import java.time.temporal.TemporalUnit;
import java.time.temporal.UnsupportedTemporalTypeException;
import java.util.List;
import java.util.Objects;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

@jdk.internal.ValueBased
public final class Period implements ChronoPeriod, Serializable {

    public static final Period ZERO;

    public static Period ofYears(int years);

    public static Period ofMonths(int months);

    public static Period ofWeeks(int weeks);

    public static Period ofDays(int days);

    public static Period of(int years, int months, int days);

    public static Period from(TemporalAmount amount);

    public static Period parse(CharSequence text);

    public static Period between(LocalDate startDateInclusive, LocalDate endDateExclusive);

    @Override
    public long get(TemporalUnit unit);

    @Override
    public List<TemporalUnit> getUnits();

    @Override
    public IsoChronology getChronology();

    public boolean isZero();

    public boolean isNegative();

    public int getYears();

    public int getMonths();

    public int getDays();

    public Period withYears(int years);

    public Period withMonths(int months);

    public Period withDays(int days);

    public Period plus(TemporalAmount amountToAdd);

    public Period plusYears(long yearsToAdd);

    public Period plusMonths(long monthsToAdd);

    public Period plusDays(long daysToAdd);

    public Period minus(TemporalAmount amountToSubtract);

    public Period minusYears(long yearsToSubtract);

    public Period minusMonths(long monthsToSubtract);

    public Period minusDays(long daysToSubtract);

    public Period multipliedBy(int scalar);

    public Period negated();

    public Period normalized();

    public long toTotalMonths();

    @Override
    public Temporal addTo(Temporal temporal);

    @Override
    public Temporal subtractFrom(Temporal temporal);

    @Override
    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object obj);

    @Override
    public int hashCode();

    @Override
    public String toString();

    void writeExternal(DataOutput out) throws IOException;

    static Period readExternal(DataInput in) throws IOException;
}
