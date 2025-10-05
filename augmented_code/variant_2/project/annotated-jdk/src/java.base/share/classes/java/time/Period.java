/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 2012, 2020, Oracle and/or its affiliates. All rights reserved.
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
package java.time;

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
import static java.time.temporal.ChronoUnit.DAYS;
    @Positive
import static java.time.temporal.ChronoUnit.MONTHS;
    @Positive
import static java.time.temporal.ChronoUnit.YEARS;
    @Positive
import java.io.DataInput;
    @Positive
import java.io.DataOutput;
    @Positive
import java.io.IOException;
    @Positive
import java.io.InvalidObjectException;
    @Positive
import java.io.ObjectInputStream;
    @Positive
import java.io.Serializable;
    @Positive
import java.time.chrono.ChronoLocalDate;
    @Positive
import java.time.chrono.ChronoPeriod;
    @Positive
import java.time.chrono.Chronology;
    @Positive
import java.time.chrono.IsoChronology;
    @Positive
import java.time.format.DateTimeParseException;
    @Positive
import java.time.temporal.ChronoUnit;
    @Positive
import java.time.temporal.Temporal;
    @Positive
import java.time.temporal.TemporalAccessor;
    @Positive
import java.time.temporal.TemporalAmount;
    @Positive
import java.time.temporal.TemporalQueries;
    @Positive
import java.time.temporal.TemporalUnit;
    @Positive
import java.time.temporal.UnsupportedTemporalTypeException;
    @Positive
import java.util.List;
    @Positive
import java.util.Objects;
    @Positive
import java.util.regex.Matcher;
    @Positive
import java.util.regex.Pattern;

    @Positive
@jdk.internal.ValueBased
    @Positive
public final class Period implements ChronoPeriod, Serializable {

    @Positive
    public static final Period ZERO;

    @Positive
    public static Period ofYears(int years);

    @Positive
    public static Period ofMonths(int months);

    @Positive
    public static Period ofWeeks(int weeks);

    @Positive
    public static Period ofDays(int days);

    @Positive
    public static Period of(int years, int months, int days);

    @Positive
    public static Period from(TemporalAmount amount);

    @Positive
    public static Period parse(CharSequence text);

    @Positive
    public static Period between(LocalDate startDateInclusive, LocalDate endDateExclusive);

    @Positive
    @Override
    @Positive
    public long get(TemporalUnit unit);

    @Positive
    @Override
    @Positive
    public List<TemporalUnit> getUnits();

    @Positive
    @Override
    @Positive
    public IsoChronology getChronology();

    @Positive
    public boolean isZero();

    @Positive
    public boolean isNegative();

    @Positive
    public int getYears();

    @Positive
    public int getMonths();

    @Positive
    public int getDays();

    @Positive
    public Period withYears(int years);

    @Positive
    public Period withMonths(int months);

    @Positive
    public Period withDays(int days);

    @Positive
    public Period plus(TemporalAmount amountToAdd);

    @Positive
    public Period plusYears(long yearsToAdd);

    @Positive
    public Period plusMonths(long monthsToAdd);

    @Positive
    public Period plusDays(long daysToAdd);

    @Positive
    public Period minus(TemporalAmount amountToSubtract);

    @Positive
    public Period minusYears(long yearsToSubtract);

    @Positive
    public Period minusMonths(long monthsToSubtract);

    @Positive
    public Period minusDays(long daysToSubtract);

    @Positive
    public Period multipliedBy(int scalar);

    @Positive
    public Period negated();

    @Positive
    public Period normalized();

    @Positive
    public long toTotalMonths();

    @Positive
    @Override
    @Positive
    public Temporal addTo(Temporal temporal);

    @Positive
    @Override
    @Positive
    public Temporal subtractFrom(Temporal temporal);

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
    void writeExternal(DataOutput out) throws IOException;

    @Positive
    static Period readExternal(DataInput in) throws IOException;
    @Positive
}
