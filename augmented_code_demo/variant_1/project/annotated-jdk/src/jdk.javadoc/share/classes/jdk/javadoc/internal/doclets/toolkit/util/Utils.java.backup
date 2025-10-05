/*
 * Copyright (c) 1999, 2021, Oracle and/or its affiliates. All rights reserved.
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
package jdk.javadoc.internal.doclets.toolkit.util;

import org.checkerframework.dataflow.qual.Pure;
import java.lang.annotation.Documented;
import java.lang.ref.SoftReference;
import java.net.URI;
import java.text.CollationKey;
import java.text.Collator;
import java.text.ParseException;
import java.text.RuleBasedCollator;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Deque;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Objects;
import java.util.Set;
import java.util.SortedSet;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.function.Predicate;
import java.util.stream.Collectors;
import javax.lang.model.AnnotatedConstruct;
import javax.lang.model.SourceVersion;
import javax.lang.model.element.AnnotationMirror;
import javax.lang.model.element.AnnotationValue;
import javax.lang.model.element.Element;
import javax.lang.model.element.ElementKind;
import javax.lang.model.element.ExecutableElement;
import javax.lang.model.element.Modifier;
import javax.lang.model.element.ModuleElement;
import javax.lang.model.element.ModuleElement.RequiresDirective;
import javax.lang.model.element.PackageElement;
import javax.lang.model.element.RecordComponentElement;
import javax.lang.model.element.TypeElement;
import javax.lang.model.element.TypeParameterElement;
import javax.lang.model.element.VariableElement;
import javax.lang.model.type.ArrayType;
import javax.lang.model.type.DeclaredType;
import javax.lang.model.type.ErrorType;
import javax.lang.model.type.ExecutableType;
import javax.lang.model.type.NoType;
import javax.lang.model.type.PrimitiveType;
import javax.lang.model.type.TypeMirror;
import javax.lang.model.type.TypeVariable;
import javax.lang.model.type.WildcardType;
import javax.lang.model.util.ElementFilter;
import javax.lang.model.util.Elements;
import javax.lang.model.util.SimpleAnnotationValueVisitor14;
import javax.lang.model.util.SimpleElementVisitor14;
import javax.lang.model.util.SimpleTypeVisitor14;
import javax.lang.model.util.TypeKindVisitor9;
import javax.lang.model.util.Types;
import javax.tools.FileObject;
import javax.tools.JavaFileManager;
import javax.tools.JavaFileManager.Location;
import javax.tools.StandardLocation;
import com.sun.source.doctree.BlockTagTree;
import com.sun.source.doctree.DeprecatedTree;
import com.sun.source.doctree.DocCommentTree;
import com.sun.source.doctree.DocTree;
import com.sun.source.doctree.DocTree.Kind;
import com.sun.source.doctree.EndElementTree;
import com.sun.source.doctree.ParamTree;
import com.sun.source.doctree.ProvidesTree;
import com.sun.source.doctree.ReturnTree;
import com.sun.source.doctree.SeeTree;
import com.sun.source.doctree.SerialDataTree;
import com.sun.source.doctree.SerialFieldTree;
import com.sun.source.doctree.SerialTree;
import com.sun.source.doctree.StartElementTree;
import com.sun.source.doctree.TextTree;
import com.sun.source.doctree.ThrowsTree;
import com.sun.source.doctree.UnknownBlockTagTree;
import com.sun.source.doctree.UsesTree;
import com.sun.source.tree.CompilationUnitTree;
import com.sun.source.tree.LineMap;
import com.sun.source.util.DocSourcePositions;
import com.sun.source.util.DocTrees;
import com.sun.source.util.TreePath;
import com.sun.tools.javac.model.JavacTypes;
import jdk.javadoc.internal.doclets.toolkit.BaseConfiguration;
import jdk.javadoc.internal.doclets.toolkit.BaseOptions;
import jdk.javadoc.internal.doclets.toolkit.CommentUtils;
import jdk.javadoc.internal.doclets.toolkit.CommentUtils.DocCommentInfo;
import jdk.javadoc.internal.doclets.toolkit.Resources;
import jdk.javadoc.internal.doclets.toolkit.taglets.BaseTaglet;
import jdk.javadoc.internal.doclets.toolkit.taglets.Taglet;
import jdk.javadoc.internal.tool.DocEnvImpl;
import static javax.lang.model.element.ElementKind.*;
import static javax.lang.model.type.TypeKind.*;
import static com.sun.source.doctree.DocTree.Kind.*;
import static jdk.javadoc.internal.doclets.toolkit.builders.ConstantsSummaryBuilder.MAX_CONSTANT_VALUE_INDEX_LENGTH;

public class Utils {

    public final BaseConfiguration configuration;

    public final DocTrees docTrees;

    public final Elements elementUtils;

    public final Types typeUtils;

    public final Comparators comparators;

    public Utils(BaseConfiguration c) {
    }

    public TypeMirror getSymbol(String signature);

    public TypeMirror getObjectType();

    public TypeMirror getExceptionType();

    public TypeMirror getErrorType();

    public TypeMirror getSerializableType();

    public TypeMirror getExternalizableType();

    public TypeMirror getIllegalArgumentExceptionType();

    public TypeMirror getNullPointerExceptionType();

    public TypeMirror getDeprecatedType();

    public TypeMirror getFunctionalInterface();

    public List<Element> excludeDeprecatedMembers(List<? extends Element> members);

    public ExecutableElement findMethod(TypeElement te, ExecutableElement method);

    @Pure
    public boolean isSubclassOf(TypeElement t1, TypeElement t2);

    public boolean executableMembersEqual(ExecutableElement e1, ExecutableElement e2);

    @Pure
    public boolean isCoreClass(TypeElement e);

    public Location getLocationForPackage(PackageElement pd);

    public Location getLocationForModule(ModuleElement mdle);

    @Pure
    public boolean isAnnotated(TypeMirror e);

    @Pure
    public boolean isAnnotated(Element e);

    @Pure
    public boolean isAnnotationType(Element e);

    @Pure
    public boolean isClass(Element e);

    @Pure
    public boolean isConstructor(Element e);

    @Pure
    public boolean isEnum(Element e);

    @Pure
    boolean isEnumConstant(Element e);

    @Pure
    public boolean isField(Element e);

    @Pure
    public boolean isInterface(Element e);

    @Pure
    public boolean isMethod(Element e);

    @Pure
    public boolean isModule(Element e);

    @Pure
    public boolean isPackage(Element e);

    @Pure
    public boolean isAbstract(Element e);

    @Pure
    public boolean isDefault(Element e);

    @Pure
    public boolean isFinal(Element e);

    @Pure
    public boolean isPackagePrivate(Element e);

    @Pure
    public boolean isPrivate(Element e);

    @Pure
    public boolean isProtected(Element e);

    @Pure
    public boolean isPublic(Element e);

    @Pure
    public boolean isProperty(String name);

    public String getPropertyName(String name);

    public String getPropertyLabel(String name);

    @Pure
    public boolean isOverviewElement(Element e);

    @Pure
    public boolean isStatic(Element e);

    @Pure
    public boolean isSerializable(TypeElement e);

    @Pure
    public boolean isExternalizable(TypeElement e);

    public boolean isRecord(TypeElement e);

    public boolean isCanonicalRecordConstructor(ExecutableElement ee);

    public SortedSet<VariableElement> serializableFields(TypeElement aclass);

    public SortedSet<ExecutableElement> serializationMethods(TypeElement aclass);

    public boolean definesSerializableFields(TypeElement aclass);

    @Pure
    public boolean isFunctionalInterface(AnnotationMirror amirror);

    @Pure
    public boolean isNoType(TypeMirror t);

    @Pure
    public boolean isOrdinaryClass(TypeElement te);

    @Pure
    public boolean isUndocumentedEnclosure(TypeElement enclosingTypeElement);

    @Pure
    public boolean isError(TypeElement te);

    @Pure
    public boolean isException(TypeElement te);

    @Pure
    public boolean isPrimitive(TypeMirror t);

    @Pure
    public boolean isExecutableElement(Element e);

    @Pure
    public boolean isVariableElement(Element e);

    @Pure
    public boolean isTypeElement(Element e);

    public String signature(ExecutableElement e, TypeElement site);

    public String flatSignature(ExecutableElement e, TypeElement site);

    public String makeSignature(ExecutableElement e, TypeElement site, boolean full);

    public String makeSignature(ExecutableElement e, TypeElement site, boolean full, boolean ignoreTypeParameters);

    public String getTypeSignature(TypeMirror t, boolean qualifiedName, boolean noTypeParameters);

    @Pure
    public boolean isArrayType(TypeMirror t);

    @Pure
    public boolean isDeclaredType(TypeMirror t);

    @Pure
    public boolean isErrorType(TypeMirror t);

    @Pure
    public boolean isIntersectionType(TypeMirror t);

    @Pure
    public boolean isTypeParameterElement(Element e);

    @Pure
    public boolean isTypeVariable(TypeMirror t);

    @Pure
    public boolean isVoid(TypeMirror t);

    @Pure
    public boolean isWildCard(TypeMirror t);

    public boolean ignoreBounds(TypeMirror bound);

    public List<? extends TypeMirror> getBounds(TypeParameterElement tpe);

    public TypeMirror getReturnType(TypeElement site, ExecutableElement ee);

    public ExecutableType asInstantiatedMethodType(TypeElement site, ExecutableElement ee);

    public TypeMirror asInstantiatedFieldType(TypeElement site, VariableElement ve);

    public TypeMirror overriddenType(ExecutableElement method);

    public TypeMirror getSuperType(TypeElement te);

    public TypeElement overriddenClass(ExecutableElement ee);

    public ExecutableElement overriddenMethod(ExecutableElement method);

    public SortedSet<TypeElement> getTypeElementsAsSortedSet(Iterable<TypeElement> typeElements);

    public List<? extends SerialDataTree> getSerialDataTrees(ExecutableElement member);

    public FileObject getFileObject(TypeElement te);

    public TypeMirror getDeclaredType(TypeElement enclosing, TypeMirror target);

    public TypeMirror getDeclaredType(Collection<TypeMirror> values, TypeElement enclosing, TypeMirror target);

    public Set<TypeMirror> getAllInterfaces(TypeElement te);

    public TypeElement findClassInPackageElement(PackageElement pkg, String className);

    public TypeElement findClass(Element element, String className);

    public String quote(String filepath);

    public String parsePackageName(PackageElement p);

    @Pure
    public boolean isDocumentedAnnotation(TypeElement annotation);

    @Pure
    public boolean isLinkable(TypeElement typeElem);

    public boolean isLinkable(TypeElement typeElem, Element elem);

    public TypeElement asTypeElement(TypeMirror t);

    public TypeMirror getComponentType(TypeMirror t);

    public String getDimension(TypeMirror t);

    public TypeElement getSuperClass(TypeElement te);

    public TypeElement getFirstVisibleSuperClassAsTypeElement(TypeElement te);

    public TypeMirror getFirstVisibleSuperClass(TypeMirror type);

    public TypeMirror getFirstVisibleSuperClass(TypeElement te);

    public String getTypeElementKindName(TypeElement te, boolean lowerCaseOnly);

    public String getTypeName(TypeMirror t, boolean fullyQualified);

    public String replaceTabs(String text);

    public CharSequence normalizeNewlines(CharSequence text);

    public static String toLowerCase(String s);

    @Pure
    public boolean isDeprecated(Element e);

    @Pure
    public boolean isDeprecatedForRemoval(Element e);

    public String getDeprecatedSince(Element e);

    public String propertyName(ExecutableElement e);

    public boolean hasHiddenTag(Element e);

    @Pure
    public boolean isSimpleOverride(ExecutableElement m);

    public SortedSet<TypeElement> filterOutPrivateClasses(Iterable<TypeElement> classlist, boolean javafx);

    public boolean elementsEqual(Element e1, Element e2);

    public int compareStrings(String s1, String s2);

    public int compareCaseCompare(String s1, String s2);

    int compareStrings(boolean caseSensitive, String s1, String s2);

    public String getHTMLTitle(Element element);

    private static class DocCollator {

        CollationKey getKey(String s);

        public int compare(String s1, String s2);
    }

    public String getQualifiedTypeName(TypeMirror t);

    public String getFullyQualifiedName(Element e);

    public String getFullyQualifiedName(Element e, final boolean outer);

    public Iterable<TypeElement> getEnclosedTypeElements(PackageElement pkg);

    public List<Element> getAnnotationMembers(TypeElement te);

    public List<TypeElement> getAnnotationTypes(PackageElement pkg);

    public List<TypeElement> getRecords(PackageElement pkg);

    public List<VariableElement> getFields(TypeElement te);

    public List<VariableElement> getFieldsUnfiltered(TypeElement te);

    public List<TypeElement> getClasses(Element e);

    public List<ExecutableElement> getConstructors(TypeElement te);

    public List<ExecutableElement> getMethods(TypeElement te);

    public int getOrdinalValue(VariableElement member);

    public Map<ModuleElement, Set<PackageElement>> getModulePackageMap();

    public Map<ModuleElement, String> getDependentModules(ModuleElement mdle);

    public String getModifiers(RequiresDirective rd);

    public long getLineNumber(Element e);

    public List<TypeElement> getInterfaces(PackageElement pkg);

    public List<VariableElement> getEnumConstants(TypeElement te);

    public List<TypeElement> getEnums(PackageElement pkg);

    public SortedSet<TypeElement> getAllClassesUnfiltered(PackageElement pkg);

    public SortedSet<TypeElement> getAllClasses(PackageElement pkg);

    public List<TypeElement> getOrdinaryClasses(Element e);

    public List<TypeElement> getErrors(Element e);

    public List<TypeElement> getExceptions(Element e);

    public boolean shouldDocument(Element e);

    public String getSimpleName(Element e);

    public TypeElement getEnclosingTypeElement(Element e);

    public String constantValueExpression(VariableElement ve);

    private static class ConstantValueExpression extends TypeKindVisitor9<String, Object> {

        @Override
        public String visitPrimitiveAsBoolean(PrimitiveType t, Object val);

        @Override
        public String visitPrimitiveAsByte(PrimitiveType t, Object val);

        @Override
        public String visitPrimitiveAsChar(PrimitiveType t, Object val);

        @Override
        public String visitPrimitiveAsDouble(PrimitiveType t, Object val);

        @Override
        public String visitPrimitiveAsFloat(PrimitiveType t, Object val);

        @Override
        public String visitPrimitiveAsLong(PrimitiveType t, Object val);

        @Override
        protected String defaultAction(TypeMirror e, Object val);
    }

    @Pure
    public boolean isEnclosingPackageIncluded(TypeElement te);

    @Pure
    public boolean isIncluded(Element e);

    @Pure
    public boolean isSpecified(Element e);

    public String getPackageName(PackageElement pkg);

    public String getModuleName(ModuleElement mdle);

    @Pure
    public boolean isAttribute(DocTree doctree);

    @Pure
    public boolean isAuthor(DocTree doctree);

    @Pure
    public boolean isComment(DocTree doctree);

    @Pure
    public boolean isDeprecated(DocTree doctree);

    @Pure
    public boolean isDocComment(DocTree doctree);

    @Pure
    public boolean isDocRoot(DocTree doctree);

    @Pure
    public boolean isEndElement(DocTree doctree);

    @Pure
    public boolean isEntity(DocTree doctree);

    @Pure
    public boolean isErroneous(DocTree doctree);

    @Pure
    public boolean isException(DocTree doctree);

    @Pure
    public boolean isIdentifier(DocTree doctree);

    @Pure
    public boolean isInheritDoc(DocTree doctree);

    @Pure
    public boolean isLink(DocTree doctree);

    @Pure
    public boolean isLinkPlain(DocTree doctree);

    @Pure
    public boolean isLiteral(DocTree doctree);

    @Pure
    public boolean isOther(DocTree doctree);

    @Pure
    public boolean isParam(DocTree doctree);

    @Pure
    public boolean isReference(DocTree doctree);

    @Pure
    public boolean isReturn(DocTree doctree);

    @Pure
    public boolean isSee(DocTree doctree);

    @Pure
    public boolean isSerial(DocTree doctree);

    @Pure
    public boolean isSerialData(DocTree doctree);

    @Pure
    public boolean isSerialField(DocTree doctree);

    @Pure
    public boolean isSince(DocTree doctree);

    @Pure
    public boolean isStartElement(DocTree doctree);

    @Pure
    public boolean isText(DocTree doctree);

    @Pure
    public boolean isThrows(DocTree doctree);

    @Pure
    public boolean isUnknownBlockTag(DocTree doctree);

    @Pure
    public boolean isUnknownInlineTag(DocTree doctree);

    @Pure
    public boolean isValue(DocTree doctree);

    @Pure
    public boolean isVersion(DocTree doctree);

    public CommentHelper getCommentHelper(Element element);

    public void removeCommentHelper(Element element);

    public List<? extends DocTree> getBlockTags(Element element);

    public List<? extends DocTree> getBlockTags(DocCommentTree dcTree);

    public List<? extends DocTree> getBlockTags(Element element, Predicate<DocTree> filter);

    public <T extends DocTree> List<? extends T> getBlockTags(Element element, Predicate<DocTree> filter, Class<T> tClass);

    public List<? extends DocTree> getBlockTags(Element element, DocTree.Kind kind);

    public <T extends DocTree> List<? extends T> getBlockTags(Element element, DocTree.Kind kind, Class<T> tClass);

    public List<? extends DocTree> getBlockTags(Element element, DocTree.Kind kind, DocTree.Kind altKind);

    public List<? extends DocTree> getBlockTags(Element element, Taglet taglet);

    public boolean hasBlockTag(Element element, DocTree.Kind kind);

    public boolean hasBlockTag(Element element, DocTree.Kind kind, final String tagName);

    boolean hasBlockTagUnchecked(Element element, DocTree.Kind kind);

    public TreePath getTreePath(Element e);

    public boolean hasDocCommentTree(Element element);

    public DocCommentTree getDocCommentTree0(Element element);

    public void checkJavaScriptInOption(String name, String value);

    public DocCommentTree getDocCommentTree(Element element);

    public List<? extends DocTree> getPreamble(Element element);

    public List<? extends DocTree> getFullBody(Element element);

    public List<? extends DocTree> getBody(Element element);

    public List<? extends DeprecatedTree> getDeprecatedTrees(Element element);

    public List<? extends ProvidesTree> getProvidesTrees(Element element);

    public List<? extends SeeTree> getSeeTrees(Element element);

    public List<? extends SerialTree> getSerialTrees(Element element);

    public List<? extends SerialFieldTree> getSerialFieldTrees(VariableElement field);

    public List<? extends ThrowsTree> getThrowsTrees(Element element);

    public List<? extends ParamTree> getTypeParamTrees(Element element);

    public List<? extends ParamTree> getParamTrees(Element element);

    public List<? extends ReturnTree> getReturnTrees(Element element);

    public List<? extends UsesTree> getUsesTrees(Element element);

    public List<? extends DocTree> getFirstSentenceTrees(Element element);

    public ModuleElement containingModule(Element e);

    public PackageElement containingPackage(Element e);

    public TypeElement getTopMostContainingTypeElement(Element e);

    private static class CommentHelperCache {

        public CommentHelperCache(Utils utils) {
        }

        public CommentHelper remove(Element key);

        public CommentHelper put(Element key, CommentHelper value);

        public CommentHelper get(Object key);

        public CommentHelper computeIfAbsent(Element key);
    }

    public static class Pair<K, L> {

        public final K first;

        public final L second;

        public Pair(K first, L second) {
        }

        @Override
        public String toString();
    }

    public Set<DeclarationPreviewLanguageFeatures> previewLanguageFeaturesUsed(Element e);

    public enum DeclarationPreviewLanguageFeatures {

        NONE(List.of(""));

        public final List<String> features;
    }

    public PreviewSummary declaredUsingPreviewAPIs(Element el);

    public static final class PreviewSummary {

        public final Set<TypeElement> previewAPI;

        public final Set<TypeElement> reflectivePreviewAPI;

        public final Set<TypeElement> declaredUsingPreviewFeature;

        public PreviewSummary(Set<TypeElement> previewAPI, Set<TypeElement> reflectivePreviewAPI, Set<TypeElement> declaredUsingPreviewFeature) {
        }

        @Override
        public String toString();
    }

    public boolean isPreviewAPI(Element el);

    public boolean isReflectivePreviewAPI(Element el);

    public Set<ElementFlag> elementFlags(Element el);

    public enum ElementFlag {

        DEPRECATED, PREVIEW
    }
}
