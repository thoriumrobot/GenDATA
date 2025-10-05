/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Copyright * Positive (c) 2000, 2019, Oracle and/or its affiliates. All rights reserved.
    @DO * Positive NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
    @Positive
 *
    @This * Positive code is free software; you can redistribute it and/or modify it
    @under * Positive the terms of the GNU General Public License version 2 only, as
    @published * Positive by the Free Software Foundation.  Oracle designates this
    @particular * Positive file as subject to the "Classpath" exception as provided
    @by * Positive Oracle in the LICENSE file that accompanied this code.
    @Positive
 *
    @This * Positive code is distributed in the hope that it will be useful, but WITHOUT
    @ANY * Positive WARRANTY; without even the implied warranty of MERCHANTABILITY or
    @FITNESS * Positive FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    @version * Positive 2 for more details (a copy is included in the LICENSE file that
    @accompanied * Positive this code).
    @Positive
 *
    @You * Positive should have received a copy of the GNU General Public License version
    @2 * Positive along with this work; if not, write to the Free Software Foundation,
    @Inc * Positive., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
    @Positive
 *
    @Please * Positive contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
    @or * Positive visit www.oracle.com if you need additional information or have any
    @questions * Positive.
    @Positive
 */
    @Positive
package javax.management.openmbean;

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
import java.util.Set;
    @Positive
import javax.management.Descriptor;
    @Positive
import javax.management.DescriptorRead;
    @Positive
import javax.management.ImmutableDescriptor;
    @Positive
import javax.management.MBeanParameterInfo;
    @Positive
import static javax.management.openmbean.OpenMBeanAttributeInfoSupport.*;

    @Positive
public class OpenMBeanParameterInfoSupport extends MBeanParameterInfo implements OpenMBeanParameterInfo {

    @Positive
    public OpenMBeanParameterInfoSupport(String name, String description, OpenType<?> openType) {
    @Positive
    }

    @Positive
    public OpenMBeanParameterInfoSupport(String name, String description, OpenType<?> openType, Descriptor descriptor) {
    @Positive
    }

    @Positive
    public <T> OpenMBeanParameterInfoSupport(String name, String description, OpenType<T> openType, T defaultValue) throws OpenDataException {
    @Positive
    }

    @Positive
    public <T> OpenMBeanParameterInfoSupport(String name, String description, OpenType<T> openType, T defaultValue, T[] legalValues) throws OpenDataException {
    @Positive
    }

    @Positive
    public <T> OpenMBeanParameterInfoSupport(String name, String description, OpenType<T> openType, T defaultValue, Comparable<T> minValue, Comparable<T> maxValue) throws OpenDataException {
    @Positive
    }

    @Positive
    public OpenType<?> getOpenType();

    @Positive
    public Object getDefaultValue();

    @Positive
    public Set<?> getLegalValues();

    @Positive
    public Comparable<?> getMinValue();

    @Positive
    public Comparable<?> getMaxValue();

    @Positive
    public boolean hasDefaultValue();

    @Positive
    public boolean hasLegalValues();

    @Positive
    public boolean hasMinValue();

    @Positive
    public boolean hasMaxValue();

    @Positive
    public boolean isValue(Object obj);

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object obj);

    @Positive
    public int hashCode();

    @Positive
    public String toString();
    @Positive
}
