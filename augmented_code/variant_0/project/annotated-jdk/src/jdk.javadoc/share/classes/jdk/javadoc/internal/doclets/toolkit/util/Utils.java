/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 1999, 2021, Oracle and/or its affiliates. All rights reserved.
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
package jdk.javadoc.internal.doclets.toolkit.util;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import java.lang.annotation.Documented;
    @Positive
import java.lang.ref.SoftReference;
    @Positive
import java.net.URI;
    @Positive
import java.text.CollationKey;
    @Positive
import java.text.Collator;
    @Positive
import java.text.ParseException;
    @Positive
import java.text.RuleBasedCollator;
    @Positive
import java.util.ArrayDeque;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.Collection;
    @Positive
import java.util.Collections;
    @Positive
import java.util.Deque;
    @Positive
import java.util.EnumSet;
    @Positive
import java.util.HashMap;
    @Positive
import java.util.HashSet;
    @Positive
import java.util.Iterator;
    @Positive
import java.util.LinkedHashMap;
    @Positive
import java.util.LinkedHashSet;
    @Positive
import java.util.List;
    @Positive
import java.util.Locale;
    @Positive
import java.util.Map;
    @Positive
import java.util.Map.Entry;
    @Positive
import java.util.Objects;
    @Positive
import java.util.Set;
    @Positive
import java.util.SortedSet;
    @Positive
import java.util.TreeMap;
    @Positive
import java.util.TreeSet;
    @Positive
import java.util.function.Predicate;
    @Positive
import java.util.stream.Collectors;
    @Positive
import javax.lang.model.AnnotatedConstruct;
    @Positive
import javax.lang.model.SourceVersion;
    @Positive
import javax.lang.model.element.AnnotationMirror;
    @Positive
import javax.lang.model.element.AnnotationValue;
    @Positive
import javax.lang.model.element.Element;
    @Positive
import javax.lang.model.element.ElementKind;
    @Positive
import javax.lang.model.element.ExecutableElement;
    @Positive
import javax.lang.model.element.Modifier;
    @Positive
import javax.lang.model.element.ModuleElement;
    @Positive
import javax.lang.model.element.ModuleElement.RequiresDirective;
    @Positive
import javax.lang.model.element.PackageElement;
    @Positive
import javax.lang.model.element.RecordComponentElement;
    @Positive
import javax.lang.model.element.TypeElement;
    @Positive
import javax.lang.model.element.TypeParameterElement;
    @Positive
import javax.lang.model.element.VariableElement;
    @Positive
import javax.lang.model.type.ArrayType;
    @Positive
import javax.lang.model.type.DeclaredType;
    @Positive
import javax.lang.model.type.ErrorType;
    @Positive
import javax.lang.model.type.ExecutableType;
    @Positive
import javax.lang.model.type.NoType;
    @Positive
import javax.lang.model.type.PrimitiveType;
    @Positive
import javax.lang.model.type.TypeMirror;
    @Positive
import javax.lang.model.type.TypeVariable;
    @Positive
import javax.lang.model.type.WildcardType;
    @Positive
import javax.lang.model.util.ElementFilter;
    @Positive
import javax.lang.model.util.Elements;
    @Positive
import javax.lang.model.util.SimpleAnnotationValueVisitor14;
    @Positive
import javax.lang.model.util.SimpleElementVisitor14;
    @Positive
import javax.lang.model.util.SimpleTypeVisitor14;
    @Positive
import javax.lang.model.util.TypeKindVisitor9;
    @Positive
import javax.lang.model.util.Types;
    @Positive
import javax.tools.FileObject;
    @Positive
import javax.tools.JavaFileManager;
    @Positive
import javax.tools.JavaFileManager.Location;
    @Positive
import javax.tools.StandardLocation;
    @Positive
import com.sun.source.doctree.BlockTagTree;
    @Positive
import com.sun.source.doctree.DeprecatedTree;
    @Positive
import com.sun.source.doctree.DocCommentTree;
    @Positive
import com.sun.source.doctree.DocTree;
    @Positive
import com.sun.source.doctree.DocTree.Kind;
    @Positive
import com.sun.source.doctree.EndElementTree;
    @Positive
import com.sun.source.doctree.ParamTree;
    @Positive
import com.sun.source.doctree.ProvidesTree;
    @Positive
import com.sun.source.doctree.ReturnTree;
    @Positive
import com.sun.source.doctree.SeeTree;
    @Positive
import com.sun.source.doctree.SerialDataTree;
    @Positive
import com.sun.source.doctree.SerialFieldTree;
    @Positive
import com.sun.source.doctree.SerialTree;
    @Positive
import com.sun.source.doctree.StartElementTree;
    @Positive
import com.sun.source.doctree.TextTree;
    @Positive
import com.sun.source.doctree.ThrowsTree;
    @Positive
import com.sun.source.doctree.UnknownBlockTagTree;
    @Positive
import com.sun.source.doctree.UsesTree;
    @Positive
import com.sun.source.tree.CompilationUnitTree;
    @Positive
import com.sun.source.tree.LineMap;
    @Positive
import com.sun.source.util.DocSourcePositions;
    @Positive
import com.sun.source.util.DocTrees;
    @Positive
import com.sun.source.util.TreePath;
    @Positive
import com.sun.tools.javac.model.JavacTypes;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.BaseConfiguration;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.BaseOptions;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.CommentUtils;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.CommentUtils.DocCommentInfo;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.Resources;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.taglets.BaseTaglet;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.taglets.Taglet;
    @Positive
import jdk.javadoc.internal.tool.DocEnvImpl;
    @Positive
import static javax.lang.model.element.ElementKind.*;
    @Positive
import static javax.lang.model.type.TypeKind.*;
    @Positive
import static com.sun.source.doctree.DocTree.Kind.*;
    @Positive
import static jdk.javadoc.internal.doclets.toolkit.builders.ConstantsSummaryBuilder.MAX_CONSTANT_VALUE_INDEX_LENGTH;

    @Positive
public class Utils {

    @Positive
    public final BaseConfiguration configuration;

    @Positive
    public final DocTrees docTrees;

    @Positive
    public final Elements elementUtils;

    @Positive
    public final Types typeUtils;

    @Positive
    public final Comparators comparators;

    @Positive
    public Utils(BaseConfiguration c) {
    @Positive
    }

    @Positive
    public TypeMirror getSymbol(String signature);

    @Positive
    public TypeMirror getObjectType();

    @Positive
    public TypeMirror getExceptionType();

    @Positive
    public TypeMirror getErrorType();

    @Positive
    public TypeMirror getSerializableType();

    @Positive
    public TypeMirror getExternalizableType();

    @Positive
    public TypeMirror getIllegalArgumentExceptionType();

    @Positive
    public TypeMirror getNullPointerExceptionType();

    @Positive
    public TypeMirror getDeprecatedType();

    @Positive
    public TypeMirror getFunctionalInterface();

    @Positive
    public List<Element> excludeDeprecatedMembers(List<? extends Element> members);

    @Positive
    public ExecutableElement findMethod(TypeElement te, ExecutableElement method);

    @Positive
    @Pure
    @Positive
    public boolean isSubclassOf(TypeElement t1, TypeElement t2);

    @Positive
    public boolean executableMembersEqual(ExecutableElement e1, ExecutableElement e2);

    @Positive
    @Pure
    @Positive
    public boolean isCoreClass(TypeElement e);

    @Positive
    public Location getLocationForPackage(PackageElement pd);

    @Positive
    public Location getLocationForModule(ModuleElement mdle);

    @Positive
    @Pure
    @Positive
    public boolean isAnnotated(TypeMirror e);

    @Positive
    @Pure
    @Positive
    public boolean isAnnotated(Element e);

    @Positive
    @Pure
    @Positive
    public boolean isAnnotationType(Element e);

    @Positive
    @Pure
    @Positive
    public boolean isClass(Element e);

    @Positive
    @Pure
    @Positive
    public boolean isConstructor(Element e);

    @Positive
    @Pure
    @Positive
    public boolean isEnum(Element e);

    @Positive
    @Pure
    @Positive
    boolean isEnumConstant(Element e);

    @Positive
    @Pure
    @Positive
    public boolean isField(Element e);

    @Positive
    @Pure
    @Positive
    public boolean isInterface(Element e);

    @Positive
    @Pure
    @Positive
    public boolean isMethod(Element e);

    @Positive
    @Pure
    @Positive
    public boolean isModule(Element e);

    @Positive
    @Pure
    @Positive
    public boolean isPackage(Element e);

    @Positive
    @Pure
    @Positive
    public boolean isAbstract(Element e);

    @Positive
    @Pure
    @Positive
    public boolean isDefault(Element e);

    @Positive
    @Pure
    @Positive
    public boolean isFinal(Element e);

    @Positive
    @Pure
    @Positive
    public boolean isPackagePrivate(Element e);

    @Positive
    @Pure
    @Positive
    public boolean isPrivate(Element e);

    @Positive
    @Pure
    @Positive
    public boolean isProtected(Element e);

    @Positive
    @Pure
    @Positive
    public boolean isPublic(Element e);

    @Positive
    @Pure
    @Positive
    public boolean isProperty(String name);

    @Positive
    public String getPropertyName(String name);

    @Positive
    public String getPropertyLabel(String name);

    @Positive
    @Pure
    @Positive
    public boolean isOverviewElement(Element e);

    @Positive
    @Pure
    @Positive
    public boolean isStatic(Element e);

    @Positive
    @Pure
    @Positive
    public boolean isSerializable(TypeElement e);

    @Positive
    @Pure
    @Positive
    public boolean isExternalizable(TypeElement e);

    @Positive
    public boolean isRecord(TypeElement e);

    @Positive
    public boolean isCanonicalRecordConstructor(ExecutableElement ee);

    @Positive
    public SortedSet<VariableElement> serializableFields(TypeElement aclass);

    @Positive
    public SortedSet<ExecutableElement> serializationMethods(TypeElement aclass);

    @Positive
    public boolean definesSerializableFields(TypeElement aclass);

    @Positive
    @Pure
    @Positive
    public boolean isFunctionalInterface(AnnotationMirror amirror);

    @Positive
    @Pure
    @Positive
    public boolean isNoType(TypeMirror t);

    @Positive
    @Pure
    @Positive
    public boolean isOrdinaryClass(TypeElement te);

    @Positive
    @Pure
    @Positive
    public boolean isUndocumentedEnclosure(TypeElement enclosingTypeElement);

    @Positive
    @Pure
    @Positive
    public boolean isError(TypeElement te);

    @Positive
    @Pure
    @Positive
    public boolean isException(TypeElement te);

    @Positive
    @Pure
    @Positive
    public boolean isPrimitive(TypeMirror t);

    @Positive
    @Pure
    @Positive
    public boolean isExecutableElement(Element e);

    @Positive
    @Pure
    @Positive
    public boolean isVariableElement(Element e);

    @Positive
    @Pure
    @Positive
    public boolean isTypeElement(Element e);

    @Positive
    public String signature(ExecutableElement e, TypeElement site);

    @Positive
    public String flatSignature(ExecutableElement e, TypeElement site);

    @Positive
    public String makeSignature(ExecutableElement e, TypeElement site, boolean full);

    @Positive
    public String makeSignature(ExecutableElement e, TypeElement site, boolean full, boolean ignoreTypeParameters);

    @Positive
    public String getTypeSignature(TypeMirror t, boolean qualifiedName, boolean noTypeParameters);

    @Positive
    @Pure
    @Positive
    public boolean isArrayType(TypeMirror t);

    @Positive
    @Pure
    @Positive
    public boolean isDeclaredType(TypeMirror t);

    @Positive
    @Pure
    @Positive
    public boolean isErrorType(TypeMirror t);

    @Positive
    @Pure
    @Positive
    public boolean isIntersectionType(TypeMirror t);

    @Positive
    @Pure
    @Positive
    public boolean isTypeParameterElement(Element e);

    @Positive
    @Pure
    @Positive
    public boolean isTypeVariable(TypeMirror t);

    @Positive
    @Pure
    @Positive
    public boolean isVoid(TypeMirror t);

    @Positive
    @Pure
    @Positive
    public boolean isWildCard(TypeMirror t);

    @Positive
    public boolean ignoreBounds(TypeMirror bound);

    @Positive
    public List<? extends TypeMirror> getBounds(TypeParameterElement tpe);

    @Positive
    public TypeMirror getReturnType(TypeElement site, ExecutableElement ee);

    @Positive
    public ExecutableType asInstantiatedMethodType(TypeElement site, ExecutableElement ee);

    @Positive
    public TypeMirror asInstantiatedFieldType(TypeElement site, VariableElement ve);

    @Positive
    public TypeMirror overriddenType(ExecutableElement method);

    @Positive
    public TypeMirror getSuperType(TypeElement te);

    @Positive
    public TypeElement overriddenClass(ExecutableElement ee);

    @Positive
    public ExecutableElement overriddenMethod(ExecutableElement method);

    @Positive
    public SortedSet<TypeElement> getTypeElementsAsSortedSet(Iterable<TypeElement> typeElements);

    @Positive
    public List<? extends SerialDataTree> getSerialDataTrees(ExecutableElement member);

    @Positive
    public FileObject getFileObject(TypeElement te);

    @Positive
    public TypeMirror getDeclaredType(TypeElement enclosing, TypeMirror target);

    @Positive
    public TypeMirror getDeclaredType(Collection<TypeMirror> values, TypeElement enclosing, TypeMirror target);

    @Positive
    public Set<TypeMirror> getAllInterfaces(TypeElement te);

    @Positive
    public TypeElement findClassInPackageElement(PackageElement pkg, String className);

    @Positive
    public TypeElement findClass(Element element, String className);

    @Positive
    public String quote(String filepath);

    @Positive
    public String parsePackageName(PackageElement p);

    @Positive
    @Pure
    @Positive
    public boolean isDocumentedAnnotation(TypeElement annotation);

    @Positive
    @Pure
    @Positive
    public boolean isLinkable(TypeElement typeElem);

    @Positive
    public boolean isLinkable(TypeElement typeElem, Element elem);

    @Positive
    public TypeElement asTypeElement(TypeMirror t);

    @Positive
    public TypeMirror getComponentType(TypeMirror t);

    @Positive
    public String getDimension(TypeMirror t);

    @Positive
    public TypeElement getSuperClass(TypeElement te);

    @Positive
    public TypeElement getFirstVisibleSuperClassAsTypeElement(TypeElement te);

    @Positive
    public TypeMirror getFirstVisibleSuperClass(TypeMirror type);

    @Positive
    public TypeMirror getFirstVisibleSuperClass(TypeElement te);

    @Positive
    public String getTypeElementKindName(TypeElement te, boolean lowerCaseOnly);

    @Positive
    public String getTypeName(TypeMirror t, boolean fullyQualified);

    @Positive
    public String replaceTabs(String text);

    @Positive
    public CharSequence normalizeNewlines(CharSequence text);

    @Positive
    public static String toLowerCase(String s);

    @Positive
    @Pure
    @Positive
    public boolean isDeprecated(Element e);

    @Positive
    @Pure
    @Positive
    public boolean isDeprecatedForRemoval(Element e);

    @Positive
    public String getDeprecatedSince(Element e);

    @Positive
    public String propertyName(ExecutableElement e);

    @Positive
    public boolean hasHiddenTag(Element e);

    @Positive
    @Pure
    @Positive
    public boolean isSimpleOverride(ExecutableElement m);

    @Positive
    public SortedSet<TypeElement> filterOutPrivateClasses(Iterable<TypeElement> classlist, boolean javafx);

    @Positive
    public boolean elementsEqual(Element e1, Element e2);

    @Positive
    public int compareStrings(String s1, String s2);

    @Positive
    public int compareCaseCompare(String s1, String s2);

    @Positive
    int compareStrings(boolean caseSensitive, String s1, String s2);

    @Positive
    public String getHTMLTitle(Element element);

    @Positive
    private static class DocCollator {

    @Positive
        CollationKey getKey(String s);

    @Positive
        public int compare(String s1, String s2);
    @Positive
    }

    @Positive
    public String getQualifiedTypeName(TypeMirror t);

    @Positive
    public String getFullyQualifiedName(Element e);

    @Positive
    public String getFullyQualifiedName(Element e, final boolean outer);

    @Positive
    public Iterable<TypeElement> getEnclosedTypeElements(PackageElement pkg);

    @Positive
    public List<Element> getAnnotationMembers(TypeElement te);

    @Positive
    public List<TypeElement> getAnnotationTypes(PackageElement pkg);

    @Positive
    public List<TypeElement> getRecords(PackageElement pkg);

    @Positive
    public List<VariableElement> getFields(TypeElement te);

    @Positive
    public List<VariableElement> getFieldsUnfiltered(TypeElement te);

    @Positive
    public List<TypeElement> getClasses(Element e);

    @Positive
    public List<ExecutableElement> getConstructors(TypeElement te);

    @Positive
    public List<ExecutableElement> getMethods(TypeElement te);

    @Positive
    public int getOrdinalValue(VariableElement member);

    @Positive
    public Map<ModuleElement, Set<PackageElement>> getModulePackageMap();

    @Positive
    public Map<ModuleElement, String> getDependentModules(ModuleElement mdle);

    @Positive
    public String getModifiers(RequiresDirective rd);

    @Positive
    public long getLineNumber(Element e);

    @Positive
    public List<TypeElement> getInterfaces(PackageElement pkg);

    @Positive
    public List<VariableElement> getEnumConstants(TypeElement te);

    @Positive
    public List<TypeElement> getEnums(PackageElement pkg);

    @Positive
    public SortedSet<TypeElement> getAllClassesUnfiltered(PackageElement pkg);

    @Positive
    public SortedSet<TypeElement> getAllClasses(PackageElement pkg);

    @Positive
    public List<TypeElement> getOrdinaryClasses(Element e);

    @Positive
    public List<TypeElement> getErrors(Element e);

    @Positive
    public List<TypeElement> getExceptions(Element e);

    @Positive
    public boolean shouldDocument(Element e);

    @Positive
    public String getSimpleName(Element e);

    @Positive
    public TypeElement getEnclosingTypeElement(Element e);

    @Positive
    public String constantValueExpression(VariableElement ve);

    @Positive
    private static class ConstantValueExpression extends TypeKindVisitor9<String, Object> {

    @Positive
        @Override
    @Positive
        public String visitPrimitiveAsBoolean(PrimitiveType t, Object val);

    @Positive
        @Override
    @Positive
        public String visitPrimitiveAsByte(PrimitiveType t, Object val);

    @Positive
        @Override
    @Positive
        public String visitPrimitiveAsChar(PrimitiveType t, Object val);

    @Positive
        @Override
    @Positive
        public String visitPrimitiveAsDouble(PrimitiveType t, Object val);

    @Positive
        @Override
    @Positive
        public String visitPrimitiveAsFloat(PrimitiveType t, Object val);

    @Positive
        @Override
    @Positive
        public String visitPrimitiveAsLong(PrimitiveType t, Object val);

    @Positive
        @Override
    @Positive
        protected String defaultAction(TypeMirror e, Object val);
    @Positive
    }

    @Positive
    @Pure
    @Positive
    public boolean isEnclosingPackageIncluded(TypeElement te);

    @Positive
    @Pure
    @Positive
    public boolean isIncluded(Element e);

    @Positive
    @Pure
    @Positive
    public boolean isSpecified(Element e);

    @Positive
    public String getPackageName(PackageElement pkg);

    @Positive
    public String getModuleName(ModuleElement mdle);

    @Positive
    @Pure
    @Positive
    public boolean isAttribute(DocTree doctree);

    @Positive
    @Pure
    @Positive
    public boolean isAuthor(DocTree doctree);

    @Positive
    @Pure
    @Positive
    public boolean isComment(DocTree doctree);

    @Positive
    @Pure
    @Positive
    public boolean isDeprecated(DocTree doctree);

    @Positive
    @Pure
    @Positive
    public boolean isDocComment(DocTree doctree);

    @Positive
    @Pure
    @Positive
    public boolean isDocRoot(DocTree doctree);

    @Positive
    @Pure
    @Positive
    public boolean isEndElement(DocTree doctree);

    @Positive
    @Pure
    @Positive
    public boolean isEntity(DocTree doctree);

    @Positive
    @Pure
    @Positive
    public boolean isErroneous(DocTree doctree);

    @Positive
    @Pure
    @Positive
    public boolean isException(DocTree doctree);

    @Positive
    @Pure
    @Positive
    public boolean isIdentifier(DocTree doctree);

    @Positive
    @Pure
    @Positive
    public boolean isInheritDoc(DocTree doctree);

    @Positive
    @Pure
    @Positive
    public boolean isLink(DocTree doctree);

    @Positive
    @Pure
    @Positive
    public boolean isLinkPlain(DocTree doctree);

    @Positive
    @Pure
    @Positive
    public boolean isLiteral(DocTree doctree);

    @Positive
    @Pure
    @Positive
    public boolean isOther(DocTree doctree);

    @Positive
    @Pure
    @Positive
    public boolean isParam(DocTree doctree);

    @Positive
    @Pure
    @Positive
    public boolean isReference(DocTree doctree);

    @Positive
    @Pure
    @Positive
    public boolean isReturn(DocTree doctree);

    @Positive
    @Pure
    @Positive
    public boolean isSee(DocTree doctree);

    @Positive
    @Pure
    @Positive
    public boolean isSerial(DocTree doctree);

    @Positive
    @Pure
    @Positive
    public boolean isSerialData(DocTree doctree);

    @Positive
    @Pure
    @Positive
    public boolean isSerialField(DocTree doctree);

    @Positive
    @Pure
    @Positive
    public boolean isSince(DocTree doctree);

    @Positive
    @Pure
    @Positive
    public boolean isStartElement(DocTree doctree);

    @Positive
    @Pure
    @Positive
    public boolean isText(DocTree doctree);

    @Positive
    @Pure
    @Positive
    public boolean isThrows(DocTree doctree);

    @Positive
    @Pure
    @Positive
    public boolean isUnknownBlockTag(DocTree doctree);

    @Positive
    @Pure
    @Positive
    public boolean isUnknownInlineTag(DocTree doctree);

    @Positive
    @Pure
    @Positive
    public boolean isValue(DocTree doctree);

    @Positive
    @Pure
    @Positive
    public boolean isVersion(DocTree doctree);

    @Positive
    public CommentHelper getCommentHelper(Element element);

    @Positive
    public void removeCommentHelper(Element element);

    @Positive
    public List<? extends DocTree> getBlockTags(Element element);

    @Positive
    public List<? extends DocTree> getBlockTags(DocCommentTree dcTree);

    @Positive
    public List<? extends DocTree> getBlockTags(Element element, Predicate<DocTree> filter);

    @Positive
    public <T extends DocTree> List<? extends T> getBlockTags(Element element, Predicate<DocTree> filter, Class<T> tClass);

    @Positive
    public List<? extends DocTree> getBlockTags(Element element, DocTree.Kind kind);

    @Positive
    public <T extends DocTree> List<? extends T> getBlockTags(Element element, DocTree.Kind kind, Class<T> tClass);

    @Positive
    public List<? extends DocTree> getBlockTags(Element element, DocTree.Kind kind, DocTree.Kind altKind);

    @Positive
    public List<? extends DocTree> getBlockTags(Element element, Taglet taglet);

    @Positive
    public boolean hasBlockTag(Element element, DocTree.Kind kind);

    @Positive
    public boolean hasBlockTag(Element element, DocTree.Kind kind, final String tagName);

    @Positive
    boolean hasBlockTagUnchecked(Element element, DocTree.Kind kind);

    @Positive
    public TreePath getTreePath(Element e);

    @Positive
    public boolean hasDocCommentTree(Element element);

    @Positive
    public DocCommentTree getDocCommentTree0(Element element);

    @Positive
    public void checkJavaScriptInOption(String name, String value);

    @Positive
    public DocCommentTree getDocCommentTree(Element element);

    @Positive
    public List<? extends DocTree> getPreamble(Element element);

    @Positive
    public List<? extends DocTree> getFullBody(Element element);

    @Positive
    public List<? extends DocTree> getBody(Element element);

    @Positive
    public List<? extends DeprecatedTree> getDeprecatedTrees(Element element);

    @Positive
    public List<? extends ProvidesTree> getProvidesTrees(Element element);

    @Positive
    public List<? extends SeeTree> getSeeTrees(Element element);

    @Positive
    public List<? extends SerialTree> getSerialTrees(Element element);

    @Positive
    public List<? extends SerialFieldTree> getSerialFieldTrees(VariableElement field);

    @Positive
    public List<? extends ThrowsTree> getThrowsTrees(Element element);

    @Positive
    public List<? extends ParamTree> getTypeParamTrees(Element element);

    @Positive
    public List<? extends ParamTree> getParamTrees(Element element);

    @Positive
    public List<? extends ReturnTree> getReturnTrees(Element element);

    @Positive
    public List<? extends UsesTree> getUsesTrees(Element element);

    @Positive
    public List<? extends DocTree> getFirstSentenceTrees(Element element);

    @Positive
    public ModuleElement containingModule(Element e);

    @Positive
    public PackageElement containingPackage(Element e);

    @Positive
    public TypeElement getTopMostContainingTypeElement(Element e);

    @Positive
    private static class CommentHelperCache {

    @Positive
        public CommentHelperCache(Utils utils) {
    @Positive
        }

    @Positive
        public CommentHelper remove(Element key);

    @Positive
        public CommentHelper put(Element key, CommentHelper value);

    @Positive
        public CommentHelper get(Object key);

    @Positive
        public CommentHelper computeIfAbsent(Element key);
    @Positive
    }

    @Positive
    public static class Pair<K, L> {

    @Positive
        public final K first;

    @Positive
        public final L second;

    @Positive
        public Pair(K first, L second) {
    @Positive
        }

    @Positive
        @Override
    @Positive
        public String toString();
    @Positive
    }

    @Positive
    public Set<DeclarationPreviewLanguageFeatures> previewLanguageFeaturesUsed(Element e);

    @Positive
    public enum DeclarationPreviewLanguageFeatures {

    @Positive
        NONE(List.of(""));

    @Positive
        public final List<String> features;
    @Positive
    }

    @Positive
    public PreviewSummary declaredUsingPreviewAPIs(Element el);

    @Positive
    public static final class PreviewSummary {

    @Positive
        public final Set<TypeElement> previewAPI;

    @Positive
        public final Set<TypeElement> reflectivePreviewAPI;

    @Positive
        public final Set<TypeElement> declaredUsingPreviewFeature;

    @Positive
        public PreviewSummary(Set<TypeElement> previewAPI, Set<TypeElement> reflectivePreviewAPI, Set<TypeElement> declaredUsingPreviewFeature) {
    @Positive
        }

    @Positive
        @Override
    @Positive
        public String toString();
    @Positive
    }

    @Positive
    public boolean isPreviewAPI(Element el);

    @Positive
    public boolean isReflectivePreviewAPI(Element el);

    @Positive
    public Set<ElementFlag> elementFlags(Element el);

    @Positive
    public enum ElementFlag {

    @Positive
        DEPRECATED, PREVIEW
    @Positive
    }
    @Positive
}
