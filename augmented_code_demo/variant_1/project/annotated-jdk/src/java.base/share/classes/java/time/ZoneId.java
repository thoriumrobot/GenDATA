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
import java.time.format.DateTimeFormatterBuilder;
    @Positive
import java.time.format.TextStyle;
    @Positive
import java.time.temporal.TemporalAccessor;
    @Positive
import java.time.temporal.TemporalField;
    @Positive
import java.time.temporal.TemporalQueries;
    @Positive
import java.time.temporal.TemporalQuery;
    @Positive
import java.time.temporal.UnsupportedTemporalTypeException;
    @Positive
import java.time.zone.ZoneRules;
    @Positive
import java.time.zone.ZoneRulesException;
    @Positive
import java.time.zone.ZoneRulesProvider;
    @Positive
import java.util.HashSet;
    @Positive
import java.util.Locale;
    @Positive
import java.util.Map;
    @Positive
import java.util.Objects;
    @Positive
import java.util.Set;
    @Positive
import java.util.TimeZone;
    @Positive
import static java.util.Map.entry;

    @Positive
@jdk.internal.ValueBased
    @Positive
public abstract class ZoneId implements Serializable {

    @Positive
    public static final Map<String, String> SHORT_IDS;

    @Positive
    public static ZoneId systemDefault();

    @Positive
    public static Set<String> getAvailableZoneIds();

    @Positive
    public static ZoneId of(String zoneId, Map<String, String> aliasMap);

    @Positive
    public static ZoneId of(String zoneId);

    @Positive
    public static ZoneId ofOffset(String prefix, ZoneOffset offset);

    @Positive
    static ZoneId of(String zoneId, boolean checkAvailable);

    @Positive
    public static ZoneId from(TemporalAccessor temporal);

    @Positive
    public abstract String getId();

    @Positive
    public String getDisplayName(TextStyle style, Locale locale);

    @Positive
    public abstract ZoneRules getRules();

    @Positive
    public ZoneId normalized();

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
    abstract void write(DataOutput out) throws IOException;
    @Positive
}

// CFWR semantic augmentation - variant 1
