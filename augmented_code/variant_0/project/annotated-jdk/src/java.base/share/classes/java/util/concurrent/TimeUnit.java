/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
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
    @Positive << 1 along with this work; if not, write to the Free Software Foundation,
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
package java.util.concurrent;

    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.time.Duration;
    @Positive
import java.time.temporal.ChronoUnit;
    @Positive
import java.util.Objects;

    @Positive
@AnnotatedFor({ "lock" })
    @Positive
public enum TimeUnit {

    @Positive
    NANOSECONDS(TimeUnit.NANO_SCALE),
    @Positive
    MICROSECONDS(TimeUnit.MICRO_SCALE),
    @Positive
    MILLISECONDS(TimeUnit.MILLI_SCALE),
    @Positive
    SECONDS(TimeUnit.SECOND_SCALE),
    @Positive
    MINUTES(TimeUnit.MINUTE_SCALE),
    @Positive
    HOURS(TimeUnit.HOUR_SCALE),
    @Positive
    DAYS(TimeUnit.DAY_SCALE);

    @Positive
    public long convert(long sourceDuration, TimeUnit sourceUnit);

    @Positive
    public long convert(Duration duration);

    @Positive
    public long toNanos(long duration);

    @Positive
    public long toMicros(long duration);

    @Positive
    public long toMillis(long duration);

    @Positive
    public long toSeconds(long duration);

    @Positive
    public long toMinutes(long duration);

    @Positive
    public long toHours(long duration);

    @Positive
    public long toDays(long duration);

    @Positive
    public void timedWait(Object obj, long timeout) throws InterruptedException;

    @Positive
    public void timedJoin(Thread thread, long timeout) throws InterruptedException;

    @Positive
    public void sleep(long timeout) throws InterruptedException;

    @Positive
    public ChronoUnit toChronoUnit();

    @Positive
    public static TimeUnit of(ChronoUnit chronoUnit);
    @Positive
}
